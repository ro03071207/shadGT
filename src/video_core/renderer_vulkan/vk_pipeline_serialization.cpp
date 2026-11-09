// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>

#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/shader_usage_serde.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#ifdef RYTY_GPU
#include "video_core/renderer_vulkan/vk_rasterizer_native.h"
#endif

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
// Synthetic dual-source exports change the fragment shader interface. Version 13 adds the
// general swizzled factor blend, which also changes the runtime color-buffer layout.
// 17/18 (FIX-192): vertex runtime info has is_indirect; stored translations of indirect draws
// left their ids biased, and older entries could read the padding it now uses.
// 19/20 (FIX-199): cube samples with an implicit level take their gradients from the direction.
// The 96-byte buffer-offset table moves the quad sample mask and stage offset ranges.
// 23/24 (FIX-215): samplers record whether they sample a cube map.
// 33/34/35: fragment cube samples use native cube-array views (port of RyGT 0066); stored
// translations made before sampled cubes as 2D arrays of faces.
static constexpr u32 ShaderBinaryVersion = 33u;
static constexpr u32 ShaderBinaryVersionQuadSampleMask = 34u;
// 25: AMD single-sample KHR barycentrics (on for AMD GPUs) change the
// fragment interface and interpolation; read only while that option is on.
static constexpr u32 ShaderBinaryVersionSingleSampleBarycentric = 35u;
// 29/30/31 (FIX-227/228): hull shaders, whose unnormalized samples at an explicit LOD are normalized
// by that mip's dimensions. Other stages keep their versions. RYGT_UNNORMALIZED_MIP=0: hull
// shaders as before too.
static u32 ShaderBinaryVersionStep(Shader::SwStage stage) {
    static const u32 step = [] {
        const char* env = std::getenv("RYGT_UNNORMALIZED_MIP");
        return env && env[0] == '0' ? 0u : 6u;
    }();
    return stage == Shader::SwStage::TessellationControl ? step : 0u;
}
static_assert(sizeof(Shader::HwFragmentRuntimeInfo) == 248 && sizeof(Shader::RuntimeInfo) == 312,
              "Legacy metadata compatibility requires the runtime-info layout to stay unchanged");
// Stored output metadata and runtime color-buffer flags include swizzled-alpha emulation.
// Version 13: stored SRT walker code also records flattened source addresses.
// Version 14: stored specializations no longer include compile-time runtime-info changes.
// 15: shader attribute loads/stores are stored (FIX-012).
// 16: the resource usage FIX-043 keys permutations by follows the Info.
// 17 (FIX-GT7-043): the Info's FMask list has another layout, holds every read as an FMask and
// their key another meaning. Older entries are not read: their Info can't be read at this layout.
// 18 (FIX-GT7-043): an FMask's T# is kept as the way to fetch it.
static constexpr u32 ShaderMetaVersion = 18u;
} // namespace Serialization

namespace Vulkan {
std::atomic<bool> single_sample_barycentric_cache_enabled{false};

// AMD-307: hw.fs.vertex_order_check was padding before. Where the check is off (every device
// except AMD's older compiler) it is cleared on load, so old stores keep matching; where it is on,
// entries carry VertexOrderCheckVersion and older ones are not read: their vertex shaders do not
// write the position copies.
static constexpr u32 VertexOrderCheckVersion = 100u;
static std::atomic<bool> vertex_order_check_cache_enabled{false};
void SetVertexOrderCheckCacheEnabled(bool enabled) {
    vertex_order_check_cache_enabled.store(enabled, std::memory_order_relaxed);
}
void SetSingleSampleBarycentricCacheEnabled(bool enabled) {
    single_sample_barycentric_cache_enabled.store(enabled, std::memory_order_relaxed);
}

// FIX-GT7-083 (TranslatedWithFaceCubes): added to the binary version of a pixel shader translated
// without native cube sampling that samples a cube. Above every other version and its additions
// (35 + 6 + 100), so older builds don't read such an entry either.
static constexpr u32 ClassicCubeVersion = 200u;
static std::atomic<bool> native_cube_cache_enabled{false};
void SetNativeCubeCacheEnabled(bool enabled) {
    native_cube_cache_enabled.store(enabled, std::memory_order_relaxed);
}
} // namespace Vulkan

namespace Vulkan {

namespace {
// SHD-006 census: how many new translations one made before any draw, from a program's code and
// registers alone, would have matched: null sharps (2D/array float textures, unstrided raw
// buffers, plain samplers), pixel inputs in order, plain float targets with the packed export.
// Logged every 250 new translations with what the others needed.
struct TranslationCensus {
    std::mutex mutex;
    u64 total{}, ps{}, ps_default{}, vs{}, vs_default{}, other{};
    std::map<std::string, u64> reasons;
} g_census;

void CountTranslation(const Shader::Info& info, const Shader::StageSpecialization& spec) {
    std::string why;
    const auto add = [&](const char* reason) {
        if (why.find(reason) == std::string::npos) {
            why += why.empty() ? "" : "+";
            why += reason;
        }
    };
    using Type = AmdGpu::ImageType;
    for (const auto& image : spec.images) {
        if (image.type == Type::Cube) {
            add("cube");
        } else if (image.type == Type::Color3D) {
            add("3d");
        } else if (image.type == Type::Color2DMsaa || image.type == Type::Color2DMsaaArray) {
            add("msaa");
        }
        if (image.is_integer) {
            add("int-texture");
        }
        if (image.is_srgb) {
            add("srgb");
        }
        if (image.num_conversion != AmdGpu::NumberConversion::None) {
            add("tex-conversion");
        }
    }
    for (const auto& buffer : spec.buffers) {
        if (buffer.stride != 0) {
            add("stride");
        }
        if (buffer.is_formatted) {
            add("formatted-buffer");
        }
    }
    for (const auto& sampler : spec.samplers) {
        if (sampler.force_unnormalized || sampler.force_degamma) {
            add("sampler");
        }
    }
    const bool is_ps = info.sw_stage == Shader::SwStage::Fragment;
    const bool is_vs = info.sw_stage == Shader::SwStage::Vertex;
    if (is_ps) {
        const auto& fs = spec.runtime_info.hw.fs;
        for (u32 i = 0; i < fs.num_inputs && i < fs.inputs.size(); ++i) {
            if (fs.inputs[i].param_index != i || fs.inputs[i].is_default) {
                add("inputs");
                break;
            }
        }
        for (const auto& cb : fs.color_buffers) {
            if (cb.data_format == AmdGpu::DataFormat::FormatInvalid) {
                continue;
            }
            const bool plain = cb.num_format == AmdGpu::NumberFormat::Float &&
                               cb.num_conversion == AmdGpu::NumberConversion::None &&
                               cb.export_format == AmdGpu::ShaderExportFormat::ABGR_FP16 &&
                               cb.swizzle == AmdGpu::IdentityMapping && !cb.blend_self_scale &&
                               !cb.blend_swizzled_alpha && !cb.blend_swizzled_factors;
            if (!plain) {
                add("outputs");
                break;
            }
        }
    }
    if (is_vs && !spec.vs_attribs.empty()) {
        add("vertex-layout");
    }
    std::scoped_lock lock{g_census.mutex};
    ++g_census.total;
    if (is_ps) {
        ++g_census.ps;
        g_census.ps_default += why.empty() ? 1 : 0;
    } else if (is_vs) {
        ++g_census.vs;
        g_census.vs_default += why.empty() ? 1 : 0;
    } else {
        ++g_census.other;
    }
    if (!why.empty()) {
        ++g_census.reasons[(is_ps ? "ps:" : is_vs ? "vs:" : "other:") + why];
    }
    if (g_census.total % 250 == 0) {
        std::vector<std::pair<u64, std::string>> top;
        for (const auto& [reason, count] : g_census.reasons) {
            top.emplace_back(count, reason);
        }
        std::ranges::sort(top, std::greater{});
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 14); ++i) {
            text += fmt::format(" {}={}", top[i].second, top[i].first);
        }
        LOG_WARNING(Render_Vulkan,
                    "SHD-006 census: {} new translations; pixel {} ({} from code alone), vertex {} "
                    "({} from code alone), other {}; needed:{}",
                    g_census.total, g_census.ps, g_census.ps_default, g_census.vs,
                    g_census.vs_default, g_census.other, text);
    }
}
} // namespace

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    CountTranslation(info, spec); // SHD-006
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    const bool masked_fragment = spec.runtime_info.hw_stage == Shader::HwStage::Fragment &&
                                 spec.runtime_info.hw.fs.quad_sample_mask;
    const bool single_sample_barycentric =
        spec.runtime_info.hw_stage == Shader::HwStage::Fragment &&
        spec.runtime_info.hw.fs.single_sample_barycentric;
    meta.Write((single_sample_barycentric ? Serialization::ShaderBinaryVersionSingleSampleBarycentric
                : masked_fragment         ? Serialization::ShaderBinaryVersionQuadSampleMask
                                          : Serialization::ShaderBinaryVersion) +
               Serialization::ShaderBinaryVersionStep(spec.runtime_info.sw_stage) +
               (vertex_order_check_cache_enabled.load(std::memory_order_relaxed)
                    ? VertexOrderCheckVersion
                    : 0u) +
               (TranslatedWithFaceCubes(spec,
                                        native_cube_cache_enabled.load(std::memory_order_relaxed))
                    ? ClassicCubeVersion
                    : 0u)); // FIX-GT7-083

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);
    WriteResourceUsage(meta, info);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx,
                    bool* other_cube_sampling) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    // FIX-GT7-083: cubes sampled as arrays of faces; checked once the stage is read.
    const bool face_cubes = binary_version >= ClassicCubeVersion;
    if (face_cubes) {
        binary_version -= ClassicCubeVersion;
    }
    if (vertex_order_check_cache_enabled.load(std::memory_order_relaxed)) { // AMD-307
        if (binary_version < VertexOrderCheckVersion) {
            return false;
        }
        binary_version -= VertexOrderCheckVersion;
    }
    // FIX-228: hull shader translations made with the other normalization are not read (the
    // stage is checked once the specialization is read).
    const u32 stored_version = binary_version;
    if (binary_version >= Serialization::ShaderBinaryVersion + 6 &&
        binary_version <= Serialization::ShaderBinaryVersionSingleSampleBarycentric + 6) {
        binary_version -= 6;
    }
    const bool single_sample_binary =
        binary_version == Serialization::ShaderBinaryVersionSingleSampleBarycentric;
    if (single_sample_binary
            ? !Vulkan::single_sample_barycentric_cache_enabled.load(std::memory_order_relaxed)
            : binary_version != Serialization::ShaderBinaryVersion &&
                  binary_version != Serialization::ShaderBinaryVersionQuadSampleMask) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    if (stored_version - binary_version !=
        Serialization::ShaderBinaryVersionStep(spec.runtime_info.sw_stage)) {
        return false; // FIX-228
    }
    if (spec.runtime_info.hw_stage == Shader::HwStage::Fragment &&
        !vertex_order_check_cache_enabled.load(std::memory_order_relaxed)) {
        spec.runtime_info.hw.fs.vertex_order_check = false; // AMD-307
    }
    if (spec.runtime_info.hw_stage == Shader::HwStage::Fragment && !single_sample_binary) {
        // This byte was padding in every older metadata revision.
        spec.runtime_info.hw.fs.single_sample_barycentric = false;
    }
    if (binary_version == Serialization::ShaderBinaryVersion &&
        spec.runtime_info.hw_stage == Shader::HwStage::Fragment) {
        // This byte was padding before quad-mask emulation, and need not have been zero.
        spec.runtime_info.hw.fs.quad_sample_mask = false;
    }
    info.Deserialize(ar, perm_hash_ar);
    ReadResourceUsage(meta, info);
    if (!StoredCubesFit(spec, info, face_cubes,
                        native_cube_cache_enabled.load(std::memory_order_relaxed))) {
        if (other_cube_sampling) {
            *other_cube_sampling = true; // FIX-GT7-088
        }
        return false; // FIX-GT7-083: made by the other engine's cube sampling
    }
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    // FIX-012: rect/quad-list helper shaders forward the vertex outputs the fragment shader
    // reads; a preloaded shader without these flags made them forward nothing.
    info.Write(loads.flags.data(), loads.flags.size());
    info.Write(stores.flags.data(), stores.flags.size());
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar, u64 walker_key, bool has_attribute_flags) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);
    attribute_flags_known = has_attribute_flags;
    if (has_attribute_flags) {
        info.Read(loads.flags.data(), loads.flags.size());
        info.Read(stores.flags.data(), stores.flags.size());
    }

    return srt_info.Deserialize(ar, srt_walker_owner, walker_key);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar, std::shared_ptr<void>& owner, u64 walker_key) {
    Serialization::Reader srt{ar};

    PersistentSrtInfo loaded{};
    if (ar.RemainingBytes() < sizeof(loaded)) return false;
    srt.Read(&loaded, sizeof(loaded));
    loaded.walker_func = nullptr; // Serialized process addresses are never valid here.
    if (loaded.flattened_bufsize_dw < NUM_USER_DATA_REGS ||
        loaded.flattened_bufsize_dw > UINT16_MAX) return false;
    std::shared_ptr<void> next_owner;
    if (loaded.walker_func_size) {
        if (loaded.walker_func_size > ar.RemainingBytes()) return false;
        auto code = RegisterWalkerCode(ar.CurrPtr(), loaded.walker_func_size, walker_key);
        ar.Advance(loaded.walker_func_size);
        loaded.walker_func = code.function;
        next_owner = std::move(code.owner);
    }
    *this = loaded;
    owner = std::move(next_owner);

    return true;
}

#if defined(RYTY_GPU) && defined(_WIN32) && !defined(NDEBUG)
// Exercises the actual cache reader and Info ownership against the built Debug DLL.
extern "C" __declspec(dllexport) int RytySrtWalkerOwnershipSelfTest(char* report, int capacity) {
    if (!report || capacity <= 0) return -1;
    try {
        static_assert(sizeof(PersistentSrtInfo) == 24);
        static_assert(offsetof(PersistentSrtInfo, walker_func_size) == 8);
        static_assert(offsetof(PersistentSrtInfo, flattened_bufsize_dw) == 16);
        // SysV walker: copy UD[0] to flat[16] and its guest address to source[16].
        constexpr u8 bytes[]{0x8B,0x07, 0x89,0x46,0x40,
                             0x48,0x89,0xBA,0x80,0,0,0, 0xC3};
        std::array<u32, NUM_USER_DATA_REGS> data{};
        data[0] = 0x12345678;
        Info original{};
        original.user_data = data;
        auto published = RegisterWalkerCode(bytes, sizeof(bytes), 0x339);
        original.srt_info = {published.function, sizeof(bytes), NUM_USER_DATA_REGS + 1};
        original.srt_walker_owner = std::move(published.owner);
        Info copy = original;
        original.srt_walker_owner.reset();
        copy.RefreshFlatBuf();
        if (copy.flattened_ud_buf[16] != data[0] ||
            copy.flattened_ud_src[16] != reinterpret_cast<u64>(data.data())) return -2;
        Serialization::Archive archive;
        copy.srt_info.Serialize(archive);
        auto wire = archive.TakeOff();
        if (wire.size() != sizeof(PersistentSrtInfo) + sizeof(bytes) ||
            std::memcmp(wire.data(), &copy.srt_info, sizeof(PersistentSrtInfo)) != 0) return -3;
        Info loaded{};
        loaded.user_data = data;
        Serialization::Archive input{std::vector<u8>{wire}};
        if (!loaded.srt_info.Deserialize(input, loaded.srt_walker_owner, 0x339) ||
            !loaded.srt_walker_owner || loaded.srt_info.walker_func == copy.srt_info.walker_func) return -4;
        copy.srt_walker_owner.reset();
        loaded.RefreshFlatBuf();
        if (loaded.flattened_ud_buf[16] != data[0] ||
            loaded.flattened_ud_src[16] != reinterpret_cast<u64>(data.data())) return -5;
        const auto keep_function = loaded.srt_info.walker_func;
        const auto keep_owner = loaded.srt_walker_owner;
        for (int which = 0; which < 4; ++which) {
            auto damaged = wire;
            if (which == 0) damaged.resize(sizeof(PersistentSrtInfo) - 1);
            if (which == 1) damaged.resize(damaged.size() - 1);
            if (which == 2) { const size_t huge = SIZE_MAX; std::memcpy(damaged.data() + 8, &huge, sizeof(huge)); }
            if (which == 3) { const u32 huge = UINT32_MAX; std::memcpy(damaged.data() + 16, &huge, sizeof(huge)); }
            Serialization::Archive bad{std::move(damaged)};
            if (loaded.srt_info.Deserialize(bad, loaded.srt_walker_owner, 0x339) ||
                loaded.srt_info.walker_func != keep_function || loaded.srt_walker_owner != keep_owner) return -6;
            loaded.RefreshFlatBuf();
            if (loaded.flattened_ud_buf[16] != data[0]) return -7;
        }
        std::snprintf(report, static_cast<size_t>(capacity),
            "passed: legacy 24-byte wire record, owned Info copy, cached execution/address, distinct cached code, four malformed records preserve prior owner");
        return 0;
    } catch (const std::exception& e) {
        std::snprintf(report, static_cast<size_t>(capacity), "failed: %s", e.what());
        return -8;
    }
}
#endif

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (!fetch_shader_data.Empty()) {
        spec.Write(sizeof(fetch_shader_data));
        fetch_shader_data.Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader

#include <filesystem>
#include <fstream>
extern "C" __declspec(dllexport) int RytyTraceTessMeta(const char* directory, const char* destination) {
    try {
        std::string result="["; unsigned count=0;
        for(const auto& path:std::filesystem::directory_iterator(directory)) {
            if(path.path().extension()!=".meta")continue;
            std::ifstream file(path.path(),std::ios::binary);
            std::vector<u8> bytes((std::istreambuf_iterator<char>(file)),{});
            if(bytes.size()>16*1024*1024)continue;
            const u64 hashes[]={0xf8543343ull,0xeae56230ull};bool found=false;
            for(auto hash:hashes)for(size_t i=0;i+8<=bytes.size();++i){u64 value;std::memcpy(&value,bytes.data()+i,8);if(value==hash){found=true;break;}}
            if(!found)continue;
            Serialization::Archive ar{std::move(bytes)};Shader::Info info{};Shader::StageSpecialization spec{};size_t permutation{};
            if(!Vulkan::LoadShaderMeta(ar,info,spec,permutation))continue;
            if(info.pgm_hash!=hashes[0]&&info.pgm_hash!=hashes[1])continue;
            if(count++)result+=",";
            const auto stride=info.sw_stage==Shader::SwStage::TessellationEval?spec.runtime_info.sw.tes.hs_output_cp_stride:spec.runtime_info.sw.tcs.hs_output_cp_stride;
            result+=fmt::format(R"({{"file":"{}","hash":{},"stage":{},"stride":{},"pointer_register":{},"descriptor_dword":{},"flat_dwords":{},"buffers":[)",path.path().filename().string(),info.pgm_hash,u32(info.sw_stage),stride,u32(info.tess_consts_ptr_base),info.tess_consts_dword_offset,info.flattened_ud_buf.size());
            unsigned n=0;for(const auto& buffer:info.buffers){if(buffer.IsSpecial())continue;const auto sharp=buffer.GetSharp(info);std::array<u32,4> raw{};std::memcpy(raw.data(),&sharp,16);if(n++)result+=",";result+=fmt::format(R"({{"address":{},"stride":{},"records":{},"words":[{},{},{},{}]}})",u64(sharp.base_address),u32(sharp.stride),u32(sharp.num_records),raw[0],raw[1],raw[2],raw[3]);}
            result+="]}";
            auto bin=std::filesystem::path(destination).parent_path()/(path.path().filename().string()+".flat");std::ofstream flat(bin,std::ios::binary);flat.write(reinterpret_cast<const char*>(info.flattened_ud_buf.data()),info.flattened_ud_buf.size()*4);
        }
        result+="]";std::ofstream report(destination);report<<result;return count;
    }catch(...){return -1;}
}
