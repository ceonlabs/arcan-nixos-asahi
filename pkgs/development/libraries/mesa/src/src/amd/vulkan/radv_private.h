/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef RADV_PRIVATE_H
#define RADV_PRIVATE_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_VALGRIND
#include <memcheck.h>
#include <valgrind.h>
#define VG(x) x
#else
#define VG(x) ((void)0)
#endif

#include "c11/threads.h"
#ifndef _WIN32
#include <amdgpu.h>
#include <xf86drm.h>
#endif
#include "compiler/shader_enums.h"
#include "util/bitscan.h"
#include "util/detect_os.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/rwlock.h"
#include "util/xmlconfig.h"
#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_debug_report.h"
#include "vk_device.h"
#include "vk_format.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_shader_module.h"
#include "vk_texcompress_astc.h"
#include "vk_texcompress_etc2.h"
#include "vk_util.h"
#include "vk_ycbcr_conversion.h"

#include "rmv/vk_rmv_common.h"
#include "rmv/vk_rmv_tokens.h"

#include "ac_binary.h"
#include "ac_gpu_info.h"
#include "ac_shader_util.h"
#include "ac_spm.h"
#include "ac_sqtt.h"
#include "ac_surface.h"
#include "ac_vcn.h"
#include "radv_constants.h"
#include "radv_descriptor_set.h"
#include "radv_radeon_winsys.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "sid.h"

#include "radix_sort/radix_sort_vk_devaddr.h"

/* Pre-declarations needed for WSI entrypoints */
struct wl_surface;
struct wl_display;
typedef struct xcb_connection_t xcb_connection_t;
typedef uint32_t xcb_visualid_t;
typedef uint32_t xcb_window_t;

#include <vulkan/vk_android_native_buffer.h>
#include <vulkan/vk_icd.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include "radv_entrypoints.h"

#include "wsi_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Helper to determine if we should compile
 * any of the Android AHB support.
 *
 * To actually enable the ext we also need
 * the necessary kernel support.
 */
#if DETECT_OS_ANDROID && ANDROID_API_LEVEL >= 26
#define RADV_SUPPORT_ANDROID_HARDWARE_BUFFER 1
#include <vndk/hardware_buffer.h>
#else
#define RADV_SUPPORT_ANDROID_HARDWARE_BUFFER 0
#endif

#if defined(VK_USE_PLATFORM_WAYLAND_KHR) || defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR) ||   \
   defined(VK_USE_PLATFORM_DISPLAY_KHR)
#define RADV_USE_WSI_PLATFORM
#endif

#ifdef ANDROID_STRICT
#define RADV_API_VERSION VK_MAKE_VERSION(1, 1, VK_HEADER_VERSION)
#else
#define RADV_API_VERSION VK_MAKE_VERSION(1, 3, VK_HEADER_VERSION)
#endif

#ifdef _WIN32
#define RADV_SUPPORT_CALIBRATED_TIMESTAMPS 0
#else
#define RADV_SUPPORT_CALIBRATED_TIMESTAMPS 1
#endif

#ifdef _WIN32
#define radv_printflike(a, b)
#else
#define radv_printflike(a, b) __attribute__((__format__(__printf__, a, b)))
#endif

/* The "RAW" clocks on Linux are called "FAST" on FreeBSD */
#if !defined(CLOCK_MONOTONIC_RAW) && defined(CLOCK_MONOTONIC_FAST)
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC_FAST
#endif

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline uint32_t
align_u32_npot(uint32_t v, uint32_t a)
{
   return (v + a - 1) / a * a;
}

static inline uint64_t
align_u64(uint64_t v, uint64_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

/** Alignment must be a power of 2. */
static inline bool
radv_is_aligned(uintmax_t n, uintmax_t a)
{
   assert(a == (a & -a));
   return (n & (a - 1)) == 0;
}

static inline uint32_t
radv_minify(uint32_t n, uint32_t levels)
{
   if (unlikely(n == 0))
      return 0;
   else
      return MAX2(n >> levels, 1);
}

static inline int
radv_float_to_sfixed(float value, unsigned frac_bits)
{
   return value * (1 << frac_bits);
}

static inline unsigned int
radv_float_to_ufixed(float value, unsigned frac_bits)
{
   return value * (1 << frac_bits);
}

/* Whenever we generate an error, pass it through this function. Useful for
 * debugging, where we can break on it. Only call at error site, not when
 * propagating errors. Might be useful to plug in a stack trace here.
 */

struct radv_image_view;
struct radv_instance;
struct rvcn_decode_buffer_s;

/* queue types */
enum radv_queue_family {
   RADV_QUEUE_GENERAL,
   RADV_QUEUE_COMPUTE,
   RADV_QUEUE_TRANSFER,
   RADV_QUEUE_SPARSE,
   RADV_QUEUE_VIDEO_DEC,
   RADV_QUEUE_VIDEO_ENC,
   RADV_MAX_QUEUE_FAMILIES,
   RADV_QUEUE_FOREIGN = RADV_MAX_QUEUE_FAMILIES,
   RADV_QUEUE_IGNORED,
};

struct radv_binning_settings {
   unsigned context_states_per_bin;    /* allowed range: [1, 6] */
   unsigned persistent_states_per_bin; /* allowed range: [1, 32] */
   unsigned fpovs_per_batch;           /* allowed range: [0, 255], 0 = unlimited */
};

struct radv_physical_device_cache_key {
   enum radeon_family family;
   uint32_t ptr_size;

   uint32_t conformant_trunc_coord : 1;
   uint32_t clear_lds : 1;
   uint32_t cs_wave32 : 1;
   uint32_t disable_aniso_single_level : 1;
   uint32_t disable_shrink_image_store : 1;
   uint32_t disable_sinking_load_input_fs : 1;
   uint32_t dual_color_blend_by_location : 1;
   uint32_t emulate_rt : 1;
   uint32_t ge_wave32 : 1;
   uint32_t invariant_geom : 1;
   uint32_t lower_discard_to_demote : 1;
   uint32_t mesh_fast_launch_2 : 1;
   uint32_t no_fmask : 1;
   uint32_t no_ngg_gs : 1;
   uint32_t no_rt : 1;
   uint32_t ps_wave32 : 1;
   uint32_t rt_wave64 : 1;
   uint32_t split_fma : 1;
   uint32_t ssbo_non_uniform : 1;
   uint32_t tex_non_uniform : 1;
   uint32_t use_llvm : 1;
   uint32_t use_ngg : 1;
   uint32_t use_ngg_culling : 1;
};

struct radv_physical_device {
   struct vk_physical_device vk;

   struct radeon_winsys *ws;
   struct radeon_info info;
   char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
   char marketing_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
   uint8_t driver_uuid[VK_UUID_SIZE];
   uint8_t device_uuid[VK_UUID_SIZE];
   uint8_t cache_uuid[VK_UUID_SIZE];

   int local_fd;
   int master_fd;
   struct wsi_device wsi_device;

   /* Whether DCC should be enabled for MSAA textures. */
   bool dcc_msaa_allowed;

   /* Whether to enable FMASK compression for MSAA textures (GFX6-GFX10.3) */
   bool use_fmask;

   /* Whether to enable NGG. */
   bool use_ngg;

   /* Whether to enable NGG culling. */
   bool use_ngg_culling;

   /* Whether to enable NGG streamout. */
   bool use_ngg_streamout;

   /* Whether to emulate the number of primitives generated by GS. */
   bool emulate_ngg_gs_query_pipeline_stat;

   /* Whether to use GS_FAST_LAUNCH(2) for mesh shaders. */
   bool mesh_fast_launch_2;

   /* Whether to emulate mesh/task shader queries. */
   bool emulate_mesh_shader_queries;

   /* Number of threads per wave. */
   uint8_t ps_wave_size;
   uint8_t cs_wave_size;
   uint8_t ge_wave_size;
   uint8_t rt_wave_size;

   /* Maximum compute shared memory size. */
   uint32_t max_shared_size;

   /* Whether to use the LLVM compiler backend */
   bool use_llvm;

   /* Whether to emulate ETC2 image support on HW without support. */
   bool emulate_etc2;

   /* Whether to emulate ASTC image support on HW without support. */
   bool emulate_astc;

   VkPhysicalDeviceMemoryProperties memory_properties;
   enum radeon_bo_domain memory_domains[VK_MAX_MEMORY_TYPES];
   enum radeon_bo_flag memory_flags[VK_MAX_MEMORY_TYPES];
   unsigned heaps;

   /* Bitmask of memory types that use the 32-bit address space. */
   uint32_t memory_types_32bit;

#ifndef _WIN32
   int available_nodes;
   drmPciBusInfo bus_info;

   dev_t primary_devid;
   dev_t render_devid;
#endif

   nir_shader_compiler_options nir_options[MESA_VULKAN_SHADER_STAGES];

   enum radv_queue_family vk_queue_to_radv[RADV_MAX_QUEUE_FAMILIES];
   uint32_t num_queues;

   uint32_t gs_table_depth;

   struct ac_hs_info hs;
   struct ac_task_info task_info;

   struct radv_binning_settings binning_settings;

   /* Performance counters. */
   struct ac_perfcounters ac_perfcounters;

   uint32_t num_perfcounters;
   struct radv_perfcounter_desc *perfcounters;

   struct {
      unsigned data0;
      unsigned data1;
      unsigned cmd;
      unsigned cntl;
   } vid_dec_reg;
   enum amd_ip_type vid_decode_ip;
   uint32_t vid_addr_gfx_mode;
   uint32_t stream_handle_base;
   uint32_t stream_handle_counter;
   uint32_t av1_version;

   struct radv_physical_device_cache_key cache_key;
};

static inline struct radv_instance *
radv_physical_device_instance(const struct radv_physical_device *pdev)
{
   return (struct radv_instance *)pdev->vk.instance;
}

uint32_t radv_find_memory_index(const struct radv_physical_device *pdev, VkMemoryPropertyFlags flags);

VkResult create_null_physical_device(struct vk_instance *vk_instance);

VkResult create_drm_physical_device(struct vk_instance *vk_instance, struct _drmDevice *device,
                                    struct vk_physical_device **out);

void radv_physical_device_destroy(struct vk_physical_device *vk_pdev);

enum radv_trace_mode {
   /** Radeon GPU Profiler */
   RADV_TRACE_MODE_RGP = 1 << VK_TRACE_MODE_COUNT,

   /** Radeon Raytracing Analyzer */
   RADV_TRACE_MODE_RRA = 1 << (VK_TRACE_MODE_COUNT + 1),

   /** Gather context rolls of submitted command buffers */
   RADV_TRACE_MODE_CTX_ROLLS = 1 << (VK_TRACE_MODE_COUNT + 2),
};

struct radv_instance {
   struct vk_instance vk;

   VkAllocationCallbacks alloc;

   uint64_t debug_flags;
   uint64_t perftest_flags;

   struct {
      struct driOptionCache options;
      struct driOptionCache available_options;

      bool enable_mrt_output_nan_fixup;
      bool disable_tc_compat_htile_in_general;
      bool disable_shrink_image_store;
      bool disable_aniso_single_level;
      bool disable_trunc_coord;
      bool zero_vram;
      bool disable_sinking_load_input_fs;
      bool flush_before_query_copy;
      bool enable_unified_heap_on_apu;
      bool tex_non_uniform;
      bool ssbo_non_uniform;
      bool flush_before_timestamp_write;
      bool force_rt_wave64;
      bool dual_color_blend_by_location;
      bool legacy_sparse_binding;
      bool force_pstate_peak_gfx11_dgpu;
      bool clear_lds;
      bool enable_dgc;
      bool enable_khr_present_wait;
      bool report_llvm9_version_string;
      bool vk_require_etc2;
      bool vk_require_astc;
      char *app_layer;
      uint8_t override_graphics_shader_version;
      uint8_t override_compute_shader_version;
      uint8_t override_ray_tracing_shader_version;
      int override_vram_size;
      int override_uniform_offset_alignment;
   } drirc;
};

VkResult radv_init_wsi(struct radv_physical_device *pdev);
void radv_finish_wsi(struct radv_physical_device *pdev);

struct radv_shader_binary_part;

bool radv_pipeline_cache_search(struct radv_device *device, struct vk_pipeline_cache *cache,
                                struct radv_pipeline *pipeline, const unsigned char *sha1,
                                bool *found_in_application_cache);

void radv_pipeline_cache_insert(struct radv_device *device, struct vk_pipeline_cache *cache,
                                struct radv_pipeline *pipeline, const unsigned char *sha1);

struct radv_ray_tracing_pipeline;
bool radv_ray_tracing_pipeline_cache_search(struct radv_device *device, struct vk_pipeline_cache *cache,
                                            struct radv_ray_tracing_pipeline *pipeline,
                                            const VkRayTracingPipelineCreateInfoKHR *create_info);

void radv_ray_tracing_pipeline_cache_insert(struct radv_device *device, struct vk_pipeline_cache *cache,
                                            struct radv_ray_tracing_pipeline *pipeline, unsigned num_stages,
                                            const unsigned char *sha1);

nir_shader *radv_pipeline_cache_lookup_nir(struct radv_device *device, struct vk_pipeline_cache *cache,
                                           gl_shader_stage stage, const blake3_hash key);

void radv_pipeline_cache_insert_nir(struct radv_device *device, struct vk_pipeline_cache *cache, const blake3_hash key,
                                    const nir_shader *nir);

struct vk_pipeline_cache_object *radv_pipeline_cache_lookup_nir_handle(struct radv_device *device,
                                                                       struct vk_pipeline_cache *cache,
                                                                       const unsigned char *sha1);

struct vk_pipeline_cache_object *radv_pipeline_cache_nir_to_handle(struct radv_device *device,
                                                                   struct vk_pipeline_cache *cache,
                                                                   struct nir_shader *nir, const unsigned char *sha1,
                                                                   bool cached);

struct nir_shader *radv_pipeline_cache_handle_to_nir(struct radv_device *device,
                                                     struct vk_pipeline_cache_object *object);

struct radv_meta_state {
   VkAllocationCallbacks alloc;

   VkPipelineCache cache;
   uint32_t initial_cache_entries;

   /*
    * For on-demand pipeline creation, makes sure that
    * only one thread tries to build a pipeline at the same time.
    */
   mtx_t mtx;

   /**
    * Use array element `i` for images with `2^i` samples.
    */
   struct {
      VkPipeline color_pipelines[NUM_META_FS_KEYS];
   } color_clear[MAX_SAMPLES_LOG2][MAX_RTS];

   struct {
      VkPipeline depth_only_pipeline[NUM_DEPTH_CLEAR_PIPELINES];
      VkPipeline stencil_only_pipeline[NUM_DEPTH_CLEAR_PIPELINES];
      VkPipeline depthstencil_pipeline[NUM_DEPTH_CLEAR_PIPELINES];

      VkPipeline depth_only_unrestricted_pipeline[NUM_DEPTH_CLEAR_PIPELINES];
      VkPipeline stencil_only_unrestricted_pipeline[NUM_DEPTH_CLEAR_PIPELINES];
      VkPipeline depthstencil_unrestricted_pipeline[NUM_DEPTH_CLEAR_PIPELINES];
   } ds_clear[MAX_SAMPLES_LOG2];

   VkPipelineLayout clear_color_p_layout;
   VkPipelineLayout clear_depth_p_layout;
   VkPipelineLayout clear_depth_unrestricted_p_layout;

   /* Optimized compute fast HTILE clear for stencil or depth only. */
   VkPipeline clear_htile_mask_pipeline;
   VkPipelineLayout clear_htile_mask_p_layout;
   VkDescriptorSetLayout clear_htile_mask_ds_layout;

   /* Copy VRS into HTILE. */
   VkPipeline copy_vrs_htile_pipeline;
   VkPipelineLayout copy_vrs_htile_p_layout;
   VkDescriptorSetLayout copy_vrs_htile_ds_layout;

   /* Clear DCC with comp-to-single. */
   VkPipeline clear_dcc_comp_to_single_pipeline[2]; /* 0: 1x, 1: 2x/4x/8x */
   VkPipelineLayout clear_dcc_comp_to_single_p_layout;
   VkDescriptorSetLayout clear_dcc_comp_to_single_ds_layout;

   struct {
      /** Pipeline that blits from a 1D image. */
      VkPipeline pipeline_1d_src[NUM_META_FS_KEYS];

      /** Pipeline that blits from a 2D image. */
      VkPipeline pipeline_2d_src[NUM_META_FS_KEYS];

      /** Pipeline that blits from a 3D image. */
      VkPipeline pipeline_3d_src[NUM_META_FS_KEYS];

      VkPipeline depth_only_1d_pipeline;
      VkPipeline depth_only_2d_pipeline;
      VkPipeline depth_only_3d_pipeline;

      VkPipeline stencil_only_1d_pipeline;
      VkPipeline stencil_only_2d_pipeline;
      VkPipeline stencil_only_3d_pipeline;
      VkPipelineLayout pipeline_layout;
      VkDescriptorSetLayout ds_layout;
   } blit;

   struct {
      VkPipelineLayout p_layouts[5];
      VkDescriptorSetLayout ds_layouts[5];
      VkPipeline pipelines[5][NUM_META_FS_KEYS];

      VkPipeline depth_only_pipeline[5];

      VkPipeline stencil_only_pipeline[5];
   } blit2d[MAX_SAMPLES_LOG2];

   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline;
      VkPipeline pipeline_3d;
   } itob;
   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline;
      VkPipeline pipeline_3d;
   } btoi;
   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline;
   } btoi_r32g32b32;
   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline[MAX_SAMPLES_LOG2];
      VkPipeline pipeline_3d;
   } itoi;
   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline;
   } itoi_r32g32b32;
   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline[MAX_SAMPLES_LOG2];
      VkPipeline pipeline_3d;
   } cleari;
   struct {
      VkPipelineLayout img_p_layout;
      VkDescriptorSetLayout img_ds_layout;
      VkPipeline pipeline;
   } cleari_r32g32b32;
   struct {
      VkPipelineLayout p_layout;
      VkDescriptorSetLayout ds_layout;
      VkPipeline pipeline[MAX_SAMPLES_LOG2];
   } fmask_copy;

   struct {
      VkPipelineLayout p_layout;
      VkPipeline pipeline[NUM_META_FS_KEYS];
   } resolve;

   struct {
      VkDescriptorSetLayout ds_layout;
      VkPipelineLayout p_layout;
      struct {
         VkPipeline pipeline;
         VkPipeline i_pipeline;
         VkPipeline srgb_pipeline;
      } rc[MAX_SAMPLES_LOG2];

      VkPipeline depth_zero_pipeline;
      struct {
         VkPipeline average_pipeline;
         VkPipeline max_pipeline;
         VkPipeline min_pipeline;
      } depth[MAX_SAMPLES_LOG2];

      VkPipeline stencil_zero_pipeline;
      struct {
         VkPipeline max_pipeline;
         VkPipeline min_pipeline;
      } stencil[MAX_SAMPLES_LOG2];
   } resolve_compute;

   struct {
      VkDescriptorSetLayout ds_layout;
      VkPipelineLayout p_layout;

      struct {
         VkPipeline pipeline[NUM_META_FS_KEYS];
      } rc[MAX_SAMPLES_LOG2];

      VkPipeline depth_zero_pipeline;
      struct {
         VkPipeline average_pipeline;
         VkPipeline max_pipeline;
         VkPipeline min_pipeline;
      } depth[MAX_SAMPLES_LOG2];

      VkPipeline stencil_zero_pipeline;
      struct {
         VkPipeline max_pipeline;
         VkPipeline min_pipeline;
      } stencil[MAX_SAMPLES_LOG2];
   } resolve_fragment;

   struct {
      VkPipelineLayout p_layout;
      VkPipeline decompress_pipeline;
      VkPipeline resummarize_pipeline;
   } depth_decomp[MAX_SAMPLES_LOG2];

   VkDescriptorSetLayout expand_depth_stencil_compute_ds_layout;
   VkPipelineLayout expand_depth_stencil_compute_p_layout;
   VkPipeline expand_depth_stencil_compute_pipeline;

   struct {
      VkPipelineLayout p_layout;
      VkPipeline cmask_eliminate_pipeline;
      VkPipeline fmask_decompress_pipeline;
      VkPipeline dcc_decompress_pipeline;

      VkDescriptorSetLayout dcc_decompress_compute_ds_layout;
      VkPipelineLayout dcc_decompress_compute_p_layout;
      VkPipeline dcc_decompress_compute_pipeline;
   } fast_clear_flush;

   struct {
      VkPipelineLayout fill_p_layout;
      VkPipelineLayout copy_p_layout;
      VkPipeline fill_pipeline;
      VkPipeline copy_pipeline;
   } buffer;

   struct {
      VkDescriptorSetLayout ds_layout;
      VkPipelineLayout p_layout;
      VkPipeline occlusion_query_pipeline;
      VkPipeline pipeline_statistics_query_pipeline;
      VkPipeline tfb_query_pipeline;
      VkPipeline timestamp_query_pipeline;
      VkPipeline pg_query_pipeline;
      VkPipeline ms_prim_gen_query_pipeline;
   } query;

   struct {
      VkDescriptorSetLayout ds_layout;
      VkPipelineLayout p_layout;
      VkPipeline pipeline[MAX_SAMPLES_LOG2];
   } fmask_expand;

   struct {
      VkDescriptorSetLayout ds_layout;
      VkPipelineLayout p_layout;
      VkPipeline pipeline[32];
   } dcc_retile;

   struct {
      VkPipelineLayout leaf_p_layout;
      VkPipeline leaf_pipeline;
      VkPipeline leaf_updateable_pipeline;
      VkPipelineLayout morton_p_layout;
      VkPipeline morton_pipeline;
      VkPipelineLayout lbvh_main_p_layout;
      VkPipeline lbvh_main_pipeline;
      VkPipelineLayout lbvh_generate_ir_p_layout;
      VkPipeline lbvh_generate_ir_pipeline;
      VkPipelineLayout ploc_p_layout;
      VkPipeline ploc_pipeline;
      VkPipelineLayout encode_p_layout;
      VkPipeline encode_pipeline;
      VkPipeline encode_compact_pipeline;
      VkPipelineLayout header_p_layout;
      VkPipeline header_pipeline;
      VkPipelineLayout update_p_layout;
      VkPipeline update_pipeline;
      VkPipelineLayout copy_p_layout;
      VkPipeline copy_pipeline;

      struct radix_sort_vk *radix_sort;

      struct {
         VkBuffer buffer;
         VkDeviceMemory memory;
         VkAccelerationStructureKHR accel_struct;
      } null;
   } accel_struct_build;

   struct vk_texcompress_etc2_state etc_decode;

   struct vk_texcompress_astc_state *astc_decode;

   struct {
      VkDescriptorSetLayout ds_layout;
      VkPipelineLayout p_layout;
      VkPipeline pipeline;
   } dgc_prepare;
};

#define RADV_NUM_HW_CTX (RADEON_CTX_PRIORITY_REALTIME + 1)

static inline bool
radv_sparse_queue_enabled(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* Dedicated sparse queue requires VK_QUEUE_SUBMIT_MODE_THREADED, which is incompatible with
    * VK_DEVICE_TIMELINE_MODE_EMULATED. */
   return pdev->info.has_timeline_syncobj && !instance->drirc.legacy_sparse_binding;
}

static inline enum radv_queue_family
vk_queue_to_radv(const struct radv_physical_device *phys_dev, int queue_family_index)
{
   if (queue_family_index == VK_QUEUE_FAMILY_EXTERNAL || queue_family_index == VK_QUEUE_FAMILY_FOREIGN_EXT)
      return RADV_QUEUE_FOREIGN;
   if (queue_family_index == VK_QUEUE_FAMILY_IGNORED)
      return RADV_QUEUE_IGNORED;

   assert(queue_family_index < RADV_MAX_QUEUE_FAMILIES);
   return phys_dev->vk_queue_to_radv[queue_family_index];
}

enum amd_ip_type radv_queue_family_to_ring(const struct radv_physical_device *dev, enum radv_queue_family f);

static inline bool
radv_has_uvd(struct radv_physical_device *phys_dev)
{
   enum radeon_family family = phys_dev->info.family;
   /* Only support UVD on TONGA+ */
   if (family < CHIP_TONGA)
      return false;
   return phys_dev->info.ip[AMD_IP_UVD].num_queues > 0;
}

struct radv_queue_ring_info {
   uint32_t scratch_size_per_wave;
   uint32_t scratch_waves;
   uint32_t compute_scratch_size_per_wave;
   uint32_t compute_scratch_waves;
   uint32_t esgs_ring_size;
   uint32_t gsvs_ring_size;
   uint32_t attr_ring_size;
   bool tess_rings;
   bool task_rings;
   bool mesh_scratch_ring;
   bool gds;
   bool gds_oa;
   bool sample_positions;
};

struct radv_queue_state {
   enum radv_queue_family qf;
   struct radv_queue_ring_info ring_info;

   struct radeon_winsys_bo *scratch_bo;
   struct radeon_winsys_bo *descriptor_bo;
   struct radeon_winsys_bo *compute_scratch_bo;
   struct radeon_winsys_bo *esgs_ring_bo;
   struct radeon_winsys_bo *gsvs_ring_bo;
   struct radeon_winsys_bo *tess_rings_bo;
   struct radeon_winsys_bo *task_rings_bo;
   struct radeon_winsys_bo *mesh_scratch_ring_bo;
   struct radeon_winsys_bo *attr_ring_bo;
   struct radeon_winsys_bo *gds_bo;
   struct radeon_winsys_bo *gds_oa_bo;

   struct radeon_cmdbuf *initial_preamble_cs;
   struct radeon_cmdbuf *initial_full_flush_preamble_cs;
   struct radeon_cmdbuf *continue_preamble_cs;
   struct radeon_cmdbuf *gang_wait_preamble_cs;
   struct radeon_cmdbuf *gang_wait_postamble_cs;

   /* the uses_shadow_regs here will be set only for general queue */
   bool uses_shadow_regs;
   /* register state is saved in shadowed_regs buffer */
   struct radeon_winsys_bo *shadowed_regs;
   /* shadow regs preamble ib. This will be the first preamble ib.
    * This ib has the packets to start register shadowing.
    */
   struct radeon_winsys_bo *shadow_regs_ib;
   uint32_t shadow_regs_ib_size_dw;
};

struct radv_queue {
   struct vk_queue vk;
   struct radeon_winsys_ctx *hw_ctx;
   enum radeon_ctx_priority priority;
   struct radv_queue_state state;
   struct radv_queue_state *follower_state;
   struct radeon_winsys_bo *gang_sem_bo;

   uint64_t last_shader_upload_seq;
   bool sqtt_present;
};

static inline struct radv_device *
radv_queue_device(const struct radv_queue *queue)
{
   return (struct radv_device *)queue->vk.base.device;
}

int radv_queue_init(struct radv_device *device, struct radv_queue *queue, int idx,
                    const VkDeviceQueueCreateInfo *create_info,
                    const VkDeviceQueueGlobalPriorityCreateInfoKHR *global_priority);

void radv_queue_finish(struct radv_queue *queue);

enum radeon_ctx_priority radv_get_queue_global_priority(const VkDeviceQueueGlobalPriorityCreateInfoKHR *pObj);

#define RADV_BORDER_COLOR_COUNT       4096
#define RADV_BORDER_COLOR_BUFFER_SIZE (sizeof(VkClearColorValue) * RADV_BORDER_COLOR_COUNT)

struct radv_device_border_color_data {
   bool used[RADV_BORDER_COLOR_COUNT];

   struct radeon_winsys_bo *bo;
   VkClearColorValue *colors_gpu_ptr;

   /* Mutex is required to guarantee vkCreateSampler thread safety
    * given that we are writing to a buffer and checking color occupation */
   mtx_t mutex;
};

enum radv_force_vrs {
   RADV_FORCE_VRS_1x1 = 0,
   RADV_FORCE_VRS_2x2,
   RADV_FORCE_VRS_2x1,
   RADV_FORCE_VRS_1x2,
};

struct radv_notifier {
   int fd;
   int watch;
   bool quit;
   thrd_t thread;
};

struct radv_memory_trace_data {
   /* ID of the PTE update event in ftrace data */
   uint16_t ftrace_update_ptes_id;

   uint32_t num_cpus;
   int *pipe_fds;
};

struct radv_rra_accel_struct_data {
   VkEvent build_event;
   uint64_t va;
   uint64_t size;
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkAccelerationStructureTypeKHR type;
   bool is_dead;
};

void radv_destroy_rra_accel_struct_data(VkDevice device, struct radv_rra_accel_struct_data *data);

struct radv_ray_history_header {
   uint32_t offset;
   uint32_t dispatch_index;
   uint32_t submit_base_index;
};

enum radv_packed_token_type {
   radv_packed_token_end_trace,
};

struct radv_packed_token_header {
   uint32_t launch_index : 29;
   uint32_t hit : 1;
   uint32_t token_type : 2;
};

struct radv_packed_end_trace_token {
   struct radv_packed_token_header header;

   uint32_t accel_struct_lo;
   uint32_t accel_struct_hi;

   uint32_t flags : 16;
   uint32_t dispatch_index : 16;

   uint32_t sbt_offset : 4;
   uint32_t sbt_stride : 4;
   uint32_t miss_index : 16;
   uint32_t cull_mask : 8;

   float origin[3];
   float tmin;
   float direction[3];
   float tmax;

   uint32_t iteration_count : 16;
   uint32_t instance_count : 16;

   uint32_t ahit_count : 16;
   uint32_t isec_count : 16;

   uint32_t primitive_id;
   uint32_t geometry_id;

   uint32_t instance_id : 24;
   uint32_t hit_kind : 8;

   float t;
};
static_assert(sizeof(struct radv_packed_end_trace_token) == 76, "Unexpected radv_packed_end_trace_token size");

enum radv_rra_ray_history_metadata_type {
   RADV_RRA_COUNTER_INFO = 1,
   RADV_RRA_DISPATCH_SIZE = 2,
   RADV_RRA_TRAVERSAL_FLAGS = 3,
};

struct radv_rra_ray_history_metadata_info {
   enum radv_rra_ray_history_metadata_type type : 32;
   uint32_t padding;
   uint64_t size;
};

enum radv_rra_pipeline_type {
   RADV_RRA_PIPELINE_RAY_TRACING,
};

struct radv_rra_ray_history_counter {
   uint32_t dispatch_size[3];
   uint32_t hit_shader_count;
   uint32_t miss_shader_count;
   uint32_t shader_count;
   uint64_t pipeline_api_hash;
   uint32_t mode;
   uint32_t mask;
   uint32_t stride;
   uint32_t data_size;
   uint32_t lost_token_size;
   uint32_t ray_id_begin;
   uint32_t ray_id_end;
   enum radv_rra_pipeline_type pipeline_type : 32;
};

struct radv_rra_ray_history_dispatch_size {
   uint32_t size[3];
   uint32_t padding;
};

struct radv_rra_ray_history_traversal_flags {
   uint32_t box_sort_mode : 1;
   uint32_t node_ptr_flags : 1;
   uint32_t reserved : 30;
   uint32_t padding;
};

struct radv_rra_ray_history_metadata {
   struct radv_rra_ray_history_metadata_info counter_info;
   struct radv_rra_ray_history_counter counter;

   struct radv_rra_ray_history_metadata_info dispatch_size_info;
   struct radv_rra_ray_history_dispatch_size dispatch_size;

   struct radv_rra_ray_history_metadata_info traversal_flags_info;
   struct radv_rra_ray_history_traversal_flags traversal_flags;
};
static_assert(sizeof(struct radv_rra_ray_history_metadata) == 136,
              "radv_rra_ray_history_metadata does not match RRA expectations");

struct radv_rra_ray_history_data {
   struct radv_rra_ray_history_metadata metadata;
};

struct radv_rra_trace_data {
   struct hash_table *accel_structs;
   struct hash_table_u64 *accel_struct_vas;
   simple_mtx_t data_mtx;
   bool validate_as;
   bool copy_after_build;
   bool triggered;
   uint32_t copy_memory_index;

   struct util_dynarray ray_history;
   VkBuffer ray_history_buffer;
   VkDeviceMemory ray_history_memory;
   void *ray_history_data;
   uint64_t ray_history_addr;
   uint32_t ray_history_buffer_size;
   uint32_t ray_history_resolution_scale;
};

enum radv_dispatch_table {
   RADV_DEVICE_DISPATCH_TABLE,
   RADV_ANNOTATE_DISPATCH_TABLE,
   RADV_APP_DISPATCH_TABLE,
   RADV_RGP_DISPATCH_TABLE,
   RADV_RRA_DISPATCH_TABLE,
   RADV_RMV_DISPATCH_TABLE,
   RADV_CTX_ROLL_DISPATCH_TABLE,
   RADV_DISPATCH_TABLE_COUNT,
};

struct radv_layer_dispatch_tables {
   struct vk_device_dispatch_table annotate;
   struct vk_device_dispatch_table app;
   struct vk_device_dispatch_table rgp;
   struct vk_device_dispatch_table rra;
   struct vk_device_dispatch_table rmv;
   struct vk_device_dispatch_table ctx_roll;
};

enum radv_buffer_robustness {
   RADV_BUFFER_ROBUSTNESS_DISABLED,
   RADV_BUFFER_ROBUSTNESS_1, /* robustBufferAccess */
   RADV_BUFFER_ROBUSTNESS_2, /* robustBufferAccess2 */
};

struct radv_sqtt_timestamp {
   uint8_t *map;
   unsigned offset;
   uint64_t size;
   struct radeon_winsys_bo *bo;
   struct list_head list;
};

struct radv_device_cache_key {
   uint32_t disable_trunc_coord : 1;
   uint32_t image_2d_view_of_3d : 1;
   uint32_t mesh_shader_queries : 1;
   uint32_t primitives_generated_query : 1;
};

struct radv_printf_format {
   char *string;
   uint32_t divergence_mask;
   uint8_t element_sizes[32];
};

struct radv_printf_data {
   uint32_t buffer_size;
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceAddress buffer_addr;
   void *data;
   struct util_dynarray formats;
};

VkResult radv_printf_data_init(struct radv_device *device);

void radv_printf_data_finish(struct radv_device *device);

struct radv_printf_buffer_header {
   uint32_t offset;
   uint32_t size;
};

typedef struct nir_builder nir_builder;
typedef struct nir_def nir_def;

void radv_build_printf(nir_builder *b, nir_def *cond, const char *format, ...);

void radv_dump_printf_data(struct radv_device *device, FILE *out);

void radv_device_associate_nir(struct radv_device *device, nir_shader *nir);

struct radv_device {
   struct vk_device vk;

   struct radeon_winsys *ws;

   struct radv_layer_dispatch_tables layer_dispatch;

   struct radeon_winsys_ctx *hw_ctx[RADV_NUM_HW_CTX];
   struct radv_meta_state meta_state;

   struct radv_queue *queues[RADV_MAX_QUEUE_FAMILIES];
   int queue_count[RADV_MAX_QUEUE_FAMILIES];

   bool pbb_allowed;
   uint32_t scratch_waves;
   uint32_t dispatch_initiator;
   uint32_t dispatch_initiator_task;

   /* MSAA sample locations.
    * The first index is the sample index.
    * The second index is the coordinate: X, Y. */
   float sample_locations_1x[1][2];
   float sample_locations_2x[2][2];
   float sample_locations_4x[4][2];
   float sample_locations_8x[8][2];

   /* GFX7 and later */
   uint32_t gfx_init_size_dw;
   struct radeon_winsys_bo *gfx_init;

   struct radeon_winsys_bo *trace_bo;
   uint32_t *trace_id_ptr;

   /* Whether to keep shader debug info, for debugging. */
   bool keep_shader_info;

   /* Backup in-memory cache to be used if the app doesn't provide one */
   struct vk_pipeline_cache *mem_cache;

   /*
    * use different counters so MSAA MRTs get consecutive surface indices,
    * even if MASK is allocated in between.
    */
   uint32_t image_mrt_offset_counter;
   uint32_t fmask_mrt_offset_counter;

   struct list_head shader_arenas;
   struct hash_table_u64 *capture_replay_arena_vas;
   unsigned shader_arena_shift;
   uint8_t shader_free_list_mask;
   struct radv_shader_free_list shader_free_list;
   struct radv_shader_free_list capture_replay_free_list;
   struct list_head shader_block_obj_pool;
   mtx_t shader_arena_mutex;

   mtx_t shader_upload_hw_ctx_mutex;
   struct radeon_winsys_ctx *shader_upload_hw_ctx;
   VkSemaphore shader_upload_sem;
   uint64_t shader_upload_seq;
   struct list_head shader_dma_submissions;
   mtx_t shader_dma_submission_list_mutex;
   cnd_t shader_dma_submission_list_cond;

   /* Whether to DMA shaders to invisible VRAM or to upload directly through BAR. */
   bool shader_use_invisible_vram;

   /* Whether the app has enabled the robustBufferAccess/robustBufferAccess2 features. */
   enum radv_buffer_robustness buffer_robustness;

   /* Whether to inline the compute dispatch size in user sgprs. */
   bool load_grid_size_from_user_sgpr;

   /* Whether the driver uses a global BO list. */
   bool use_global_bo_list;

   /* Whether anisotropy is forced with RADV_TEX_ANISO (-1 is disabled). */
   int force_aniso;

   /* Always disable TRUNC_COORD. */
   bool disable_trunc_coord;

   struct radv_device_border_color_data border_color_data;

   /* Thread trace. */
   struct ac_sqtt sqtt;
   bool sqtt_enabled;
   bool sqtt_triggered;

   /* SQTT timestamps for queue events. */
   simple_mtx_t sqtt_timestamp_mtx;
   struct radv_sqtt_timestamp sqtt_timestamp;

   /* SQTT timed cmd buffers. */
   simple_mtx_t sqtt_command_pool_mtx;
   struct vk_command_pool *sqtt_command_pool[2];

   /* Memory trace. */
   struct radv_memory_trace_data memory_trace;

   /* SPM. */
   struct ac_spm spm;

   /* Radeon Raytracing Analyzer trace. */
   struct radv_rra_trace_data rra_trace;

   FILE *ctx_roll_file;
   simple_mtx_t ctx_roll_mtx;

   /* Trap handler. */
   struct radv_shader *trap_handler_shader;
   struct radeon_winsys_bo *tma_bo; /* Trap Memory Address */
   uint32_t *tma_ptr;

   /* Overallocation. */
   bool overallocation_disallowed;
   uint64_t allocated_memory_size[VK_MAX_MEMORY_HEAPS];
   mtx_t overallocation_mutex;

   /* RADV_FORCE_VRS. */
   struct radv_notifier notifier;
   enum radv_force_vrs force_vrs;

   /* Depth image for VRS when not bound by the app. */
   struct {
      struct radv_image *image;
      struct radv_buffer *buffer; /* HTILE */
      struct radv_device_memory *mem;
   } vrs;

   /* Prime blit sdma queue */
   struct radv_queue *private_sdma_queue;

   struct radv_shader_part_cache vs_prologs;
   struct radv_shader_part *simple_vs_prologs[MAX_VERTEX_ATTRIBS];
   struct radv_shader_part *instance_rate_vs_prologs[816];

   struct radv_shader_part_cache ps_epilogs;

   simple_mtx_t trace_mtx;

   /* Whether per-vertex VRS is forced. */
   bool force_vrs_enabled;

   simple_mtx_t pstate_mtx;
   unsigned pstate_cnt;

   /* BO to contain some performance counter helpers:
    * - A lock for profiling cmdbuffers.
    * - a temporary fence for the end query synchronization.
    * - the pass to use for profiling. (as an array of bools)
    */
   struct radeon_winsys_bo *perf_counter_bo;

   /* Interleaved lock/unlock commandbuffers for perfcounter passes. */
   struct radeon_cmdbuf **perf_counter_lock_cs;

   bool uses_shadow_regs;

   struct hash_table *rt_handles;
   simple_mtx_t rt_handles_mtx;

   struct radv_printf_data printf;

   struct radv_device_cache_key cache_key;
   blake3_hash cache_hash;

   /* Not NULL if a GPU hang report has been generated for VK_EXT_device_fault. */
   char *gpu_hang_report;

   /* For indirect compute pipeline binds with DGC only. */
   simple_mtx_t compute_scratch_mtx;
   uint32_t compute_scratch_size_per_wave;
   uint32_t compute_scratch_waves;
};

static inline struct radv_physical_device *
radv_device_physical(const struct radv_device *dev)
{
   return (struct radv_physical_device *)dev->vk.physical;
}

bool radv_device_set_pstate(struct radv_device *device, bool enable);
bool radv_device_acquire_performance_counters(struct radv_device *device);
void radv_device_release_performance_counters(struct radv_device *device);

struct radv_device_memory {
   struct vk_object_base base;
   struct radeon_winsys_bo *bo;
   /* for dedicated allocations */
   struct radv_image *image;
   struct radv_buffer *buffer;
   uint32_t heap_index;
   uint64_t alloc_size;
   void *map;
   void *user_ptr;

#if RADV_SUPPORT_ANDROID_HARDWARE_BUFFER
   struct AHardwareBuffer *android_hardware_buffer;
#endif
};

void radv_device_memory_init(struct radv_device_memory *mem, struct radv_device *device, struct radeon_winsys_bo *bo);
void radv_device_memory_finish(struct radv_device_memory *mem);


enum radv_dynamic_state_bits {
   RADV_DYNAMIC_VIEWPORT = 1ull << 0,
   RADV_DYNAMIC_SCISSOR = 1ull << 1,
   RADV_DYNAMIC_LINE_WIDTH = 1ull << 2,
   RADV_DYNAMIC_DEPTH_BIAS = 1ull << 3,
   RADV_DYNAMIC_BLEND_CONSTANTS = 1ull << 4,
   RADV_DYNAMIC_DEPTH_BOUNDS = 1ull << 5,
   RADV_DYNAMIC_STENCIL_COMPARE_MASK = 1ull << 6,
   RADV_DYNAMIC_STENCIL_WRITE_MASK = 1ull << 7,
   RADV_DYNAMIC_STENCIL_REFERENCE = 1ull << 8,
   RADV_DYNAMIC_DISCARD_RECTANGLE = 1ull << 9,
   RADV_DYNAMIC_SAMPLE_LOCATIONS = 1ull << 10,
   RADV_DYNAMIC_LINE_STIPPLE = 1ull << 11,
   RADV_DYNAMIC_CULL_MODE = 1ull << 12,
   RADV_DYNAMIC_FRONT_FACE = 1ull << 13,
   RADV_DYNAMIC_PRIMITIVE_TOPOLOGY = 1ull << 14,
   RADV_DYNAMIC_DEPTH_TEST_ENABLE = 1ull << 15,
   RADV_DYNAMIC_DEPTH_WRITE_ENABLE = 1ull << 16,
   RADV_DYNAMIC_DEPTH_COMPARE_OP = 1ull << 17,
   RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE = 1ull << 18,
   RADV_DYNAMIC_STENCIL_TEST_ENABLE = 1ull << 19,
   RADV_DYNAMIC_STENCIL_OP = 1ull << 20,
   RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE = 1ull << 21,
   RADV_DYNAMIC_FRAGMENT_SHADING_RATE = 1ull << 22,
   RADV_DYNAMIC_PATCH_CONTROL_POINTS = 1ull << 23,
   RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE = 1ull << 24,
   RADV_DYNAMIC_DEPTH_BIAS_ENABLE = 1ull << 25,
   RADV_DYNAMIC_LOGIC_OP = 1ull << 26,
   RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE = 1ull << 27,
   RADV_DYNAMIC_COLOR_WRITE_ENABLE = 1ull << 28,
   RADV_DYNAMIC_VERTEX_INPUT = 1ull << 29,
   RADV_DYNAMIC_POLYGON_MODE = 1ull << 30,
   RADV_DYNAMIC_TESS_DOMAIN_ORIGIN = 1ull << 31,
   RADV_DYNAMIC_LOGIC_OP_ENABLE = 1ull << 32,
   RADV_DYNAMIC_LINE_STIPPLE_ENABLE = 1ull << 33,
   RADV_DYNAMIC_ALPHA_TO_COVERAGE_ENABLE = 1ull << 34,
   RADV_DYNAMIC_SAMPLE_MASK = 1ull << 35,
   RADV_DYNAMIC_DEPTH_CLIP_ENABLE = 1ull << 36,
   RADV_DYNAMIC_CONSERVATIVE_RAST_MODE = 1ull << 37,
   RADV_DYNAMIC_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE = 1ull << 38,
   RADV_DYNAMIC_PROVOKING_VERTEX_MODE = 1ull << 39,
   RADV_DYNAMIC_DEPTH_CLAMP_ENABLE = 1ull << 40,
   RADV_DYNAMIC_COLOR_WRITE_MASK = 1ull << 41,
   RADV_DYNAMIC_COLOR_BLEND_ENABLE = 1ull << 42,
   RADV_DYNAMIC_RASTERIZATION_SAMPLES = 1ull << 43,
   RADV_DYNAMIC_LINE_RASTERIZATION_MODE = 1ull << 44,
   RADV_DYNAMIC_COLOR_BLEND_EQUATION = 1ull << 45,
   RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE = 1ull << 46,
   RADV_DYNAMIC_DISCARD_RECTANGLE_MODE = 1ull << 47,
   RADV_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE = 1ull << 48,
   RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE = 1ull << 49,
   RADV_DYNAMIC_ALPHA_TO_ONE_ENABLE = 1ull << 50,
   RADV_DYNAMIC_ALL = (1ull << 51) - 1,
};

enum radv_cmd_dirty_bits {
   /* Keep the dynamic state dirty bits in sync with
    * enum radv_dynamic_state_bits */
   RADV_CMD_DIRTY_DYNAMIC_VIEWPORT = 1ull << 0,
   RADV_CMD_DIRTY_DYNAMIC_SCISSOR = 1ull << 1,
   RADV_CMD_DIRTY_DYNAMIC_LINE_WIDTH = 1ull << 2,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_BIAS = 1ull << 3,
   RADV_CMD_DIRTY_DYNAMIC_BLEND_CONSTANTS = 1ull << 4,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_BOUNDS = 1ull << 5,
   RADV_CMD_DIRTY_DYNAMIC_STENCIL_COMPARE_MASK = 1ull << 6,
   RADV_CMD_DIRTY_DYNAMIC_STENCIL_WRITE_MASK = 1ull << 7,
   RADV_CMD_DIRTY_DYNAMIC_STENCIL_REFERENCE = 1ull << 8,
   RADV_CMD_DIRTY_DYNAMIC_DISCARD_RECTANGLE = 1ull << 9,
   RADV_CMD_DIRTY_DYNAMIC_SAMPLE_LOCATIONS = 1ull << 10,
   RADV_CMD_DIRTY_DYNAMIC_LINE_STIPPLE = 1ull << 11,
   RADV_CMD_DIRTY_DYNAMIC_CULL_MODE = 1ull << 12,
   RADV_CMD_DIRTY_DYNAMIC_FRONT_FACE = 1ull << 13,
   RADV_CMD_DIRTY_DYNAMIC_PRIMITIVE_TOPOLOGY = 1ull << 14,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_TEST_ENABLE = 1ull << 15,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_WRITE_ENABLE = 1ull << 16,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_COMPARE_OP = 1ull << 17,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE = 1ull << 18,
   RADV_CMD_DIRTY_DYNAMIC_STENCIL_TEST_ENABLE = 1ull << 19,
   RADV_CMD_DIRTY_DYNAMIC_STENCIL_OP = 1ull << 20,
   RADV_CMD_DIRTY_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE = 1ull << 21,
   RADV_CMD_DIRTY_DYNAMIC_FRAGMENT_SHADING_RATE = 1ull << 22,
   RADV_CMD_DIRTY_DYNAMIC_PATCH_CONTROL_POINTS = 1ull << 23,
   RADV_CMD_DIRTY_DYNAMIC_RASTERIZER_DISCARD_ENABLE = 1ull << 24,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_BIAS_ENABLE = 1ull << 25,
   RADV_CMD_DIRTY_DYNAMIC_LOGIC_OP = 1ull << 26,
   RADV_CMD_DIRTY_DYNAMIC_PRIMITIVE_RESTART_ENABLE = 1ull << 27,
   RADV_CMD_DIRTY_DYNAMIC_COLOR_WRITE_ENABLE = 1ull << 28,
   RADV_CMD_DIRTY_DYNAMIC_VERTEX_INPUT = 1ull << 29,
   RADV_CMD_DIRTY_DYNAMIC_POLYGON_MODE = 1ull << 30,
   RADV_CMD_DIRTY_DYNAMIC_TESS_DOMAIN_ORIGIN = 1ull << 31,
   RADV_CMD_DIRTY_DYNAMIC_LOGIC_OP_ENABLE = 1ull << 32,
   RADV_CMD_DIRTY_DYNAMIC_LINE_STIPPLE_ENABLE = 1ull << 33,
   RADV_CMD_DIRTY_DYNAMIC_ALPHA_TO_COVERAGE_ENABLE = 1ull << 34,
   RADV_CMD_DIRTY_DYNAMIC_SAMPLE_MASK = 1ull << 35,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_CLIP_ENABLE = 1ull << 36,
   RADV_CMD_DIRTY_DYNAMIC_CONSERVATIVE_RAST_MODE = 1ull << 37,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE = 1ull << 38,
   RADV_CMD_DIRTY_DYNAMIC_PROVOKING_VERTEX_MODE = 1ull << 39,
   RADV_CMD_DIRTY_DYNAMIC_DEPTH_CLAMP_ENABLE = 1ull << 40,
   RADV_CMD_DIRTY_DYNAMIC_COLOR_WRITE_MASK = 1ull << 41,
   RADV_CMD_DIRTY_DYNAMIC_COLOR_BLEND_ENABLE = 1ull << 42,
   RADV_CMD_DIRTY_DYNAMIC_RASTERIZATION_SAMPLES = 1ull << 43,
   RADV_CMD_DIRTY_DYNAMIC_LINE_RASTERIZATION_MODE = 1ull << 44,
   RADV_CMD_DIRTY_DYNAMIC_COLOR_BLEND_EQUATION = 1ull << 45,
   RADV_CMD_DIRTY_DYNAMIC_DISCARD_RECTANGLE_ENABLE = 1ull << 46,
   RADV_CMD_DIRTY_DYNAMIC_DISCARD_RECTANGLE_MODE = 1ull << 47,
   RADV_CMD_DIRTY_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE = 1ull << 48,
   RADV_CMD_DIRTY_DYNAMIC_SAMPLE_LOCATIONS_ENABLE = 1ull << 49,
   RADV_CMD_DIRTY_DYNAMIC_ALPHA_TO_ONE_ENABLE = 1ull << 50,
   RADV_CMD_DIRTY_DYNAMIC_ALL = (1ull << 51) - 1,
   RADV_CMD_DIRTY_PIPELINE = 1ull << 51,
   RADV_CMD_DIRTY_INDEX_BUFFER = 1ull << 52,
   RADV_CMD_DIRTY_FRAMEBUFFER = 1ull << 53,
   RADV_CMD_DIRTY_VERTEX_BUFFER = 1ull << 54,
   RADV_CMD_DIRTY_STREAMOUT_BUFFER = 1ull << 55,
   RADV_CMD_DIRTY_GUARDBAND = 1ull << 56,
   RADV_CMD_DIRTY_RBPLUS = 1ull << 57,
   RADV_CMD_DIRTY_SHADER_QUERY = 1ull << 58,
   RADV_CMD_DIRTY_OCCLUSION_QUERY = 1ull << 59,
   RADV_CMD_DIRTY_DB_SHADER_CONTROL = 1ull << 60,
   RADV_CMD_DIRTY_STREAMOUT_ENABLE = 1ull << 61,
   RADV_CMD_DIRTY_GRAPHICS_SHADERS = 1ull << 62,
};

enum radv_cmd_flush_bits {
   /* Instruction cache. */
   RADV_CMD_FLAG_INV_ICACHE = 1 << 0,
   /* Scalar L1 cache. */
   RADV_CMD_FLAG_INV_SCACHE = 1 << 1,
   /* Vector L1 cache. */
   RADV_CMD_FLAG_INV_VCACHE = 1 << 2,
   /* L2 cache + L2 metadata cache writeback & invalidate.
    * GFX6-8: Used by shaders only. GFX9-10: Used by everything. */
   RADV_CMD_FLAG_INV_L2 = 1 << 3,
   /* L2 writeback (write dirty L2 lines to memory for non-L2 clients).
    * Only used for coherency with non-L2 clients like CB, DB, CP on GFX6-8.
    * GFX6-7 will do complete invalidation, because the writeback is unsupported. */
   RADV_CMD_FLAG_WB_L2 = 1 << 4,
   /* Invalidate the metadata cache. To be used when the DCC/HTILE metadata
    * changed and we want to read an image from shaders. */
   RADV_CMD_FLAG_INV_L2_METADATA = 1 << 5,
   /* Framebuffer caches */
   RADV_CMD_FLAG_FLUSH_AND_INV_CB_META = 1 << 6,
   RADV_CMD_FLAG_FLUSH_AND_INV_DB_META = 1 << 7,
   RADV_CMD_FLAG_FLUSH_AND_INV_DB = 1 << 8,
   RADV_CMD_FLAG_FLUSH_AND_INV_CB = 1 << 9,
   /* Engine synchronization. */
   RADV_CMD_FLAG_VS_PARTIAL_FLUSH = 1 << 10,
   RADV_CMD_FLAG_PS_PARTIAL_FLUSH = 1 << 11,
   RADV_CMD_FLAG_CS_PARTIAL_FLUSH = 1 << 12,
   RADV_CMD_FLAG_VGT_FLUSH = 1 << 13,
   /* Pipeline query controls. */
   RADV_CMD_FLAG_START_PIPELINE_STATS = 1 << 14,
   RADV_CMD_FLAG_STOP_PIPELINE_STATS = 1 << 15,
   RADV_CMD_FLAG_VGT_STREAMOUT_SYNC = 1 << 16,

   RADV_CMD_FLUSH_AND_INV_FRAMEBUFFER = (RADV_CMD_FLAG_FLUSH_AND_INV_CB | RADV_CMD_FLAG_FLUSH_AND_INV_CB_META |
                                         RADV_CMD_FLAG_FLUSH_AND_INV_DB | RADV_CMD_FLAG_FLUSH_AND_INV_DB_META),

   RADV_CMD_FLUSH_ALL_COMPUTE = (RADV_CMD_FLAG_INV_ICACHE | RADV_CMD_FLAG_INV_SCACHE | RADV_CMD_FLAG_INV_VCACHE |
                                 RADV_CMD_FLAG_INV_L2 | RADV_CMD_FLAG_WB_L2 | RADV_CMD_FLAG_CS_PARTIAL_FLUSH),
};

struct radv_vertex_binding {
   VkDeviceSize offset;
   VkDeviceSize size;
   VkDeviceSize stride;
};

struct radv_streamout_binding {
   struct radv_buffer *buffer;
   VkDeviceSize offset;
   VkDeviceSize size;
};

struct radv_streamout_state {
   /* Mask of bound streamout buffers. */
   uint8_t enabled_mask;

   /* State of VGT_STRMOUT_BUFFER_(CONFIG|END) */
   uint32_t hw_enabled_mask;

   /* State of VGT_STRMOUT_(CONFIG|EN) */
   bool streamout_enabled;
};

struct radv_sample_locations_state {
   VkSampleCountFlagBits per_pixel;
   VkExtent2D grid_size;
   uint32_t count;
   VkSampleLocationEXT locations[MAX_SAMPLE_LOCATIONS];
};

struct radv_dynamic_state {
   struct vk_dynamic_graphics_state vk;

   /**
    * Bitmask of (1ull << VK_DYNAMIC_STATE_*).
    * Defines the set of saved dynamic state.
    */
   uint64_t mask;

   struct {
      struct {
         float scale[3];
         float translate[3];
      } xform[MAX_VIEWPORTS];
   } hw_vp;

   struct radv_sample_locations_state sample_location;

   VkImageAspectFlags feedback_loop_aspects;
};

const char *radv_get_debug_option_name(int id);

const char *radv_get_perftest_option_name(int id);

struct radv_color_buffer_info {
   uint64_t cb_color_base;
   uint64_t cb_color_cmask;
   uint64_t cb_color_fmask;
   uint64_t cb_dcc_base;
   uint32_t cb_color_slice;
   uint32_t cb_color_view;
   uint32_t cb_color_info;
   uint32_t cb_color_attrib;
   uint32_t cb_color_attrib2; /* GFX9 and later */
   uint32_t cb_color_attrib3; /* GFX10 and later */
   uint32_t cb_dcc_control;
   uint32_t cb_color_cmask_slice;
   uint32_t cb_color_fmask_slice;
   union {
      uint32_t cb_color_pitch; // GFX6-GFX8
      uint32_t cb_mrt_epitch;  // GFX9+
   };
};

struct radv_ds_buffer_info {
   uint64_t db_z_read_base;
   uint64_t db_stencil_read_base;
   uint64_t db_z_write_base;
   uint64_t db_stencil_write_base;
   uint64_t db_htile_data_base;
   uint32_t db_depth_info;
   uint32_t db_z_info;
   uint32_t db_stencil_info;
   uint32_t db_depth_view;
   uint32_t db_depth_size;
   uint32_t db_depth_slice;
   uint32_t db_htile_surface;
   uint32_t db_z_info2;       /* GFX9 only */
   uint32_t db_stencil_info2; /* GFX9 only */
   uint32_t db_render_override2;
   uint32_t db_render_control;
};

void radv_initialise_color_surface(struct radv_device *device, struct radv_color_buffer_info *cb,
                                   struct radv_image_view *iview);
void radv_initialise_ds_surface(const struct radv_device *device, struct radv_ds_buffer_info *ds,
                                struct radv_image_view *iview, VkImageAspectFlags ds_aspects);
void radv_initialise_vrs_surface(struct radv_image *image, struct radv_buffer *htile_buffer,
                                 struct radv_ds_buffer_info *ds);

void radv_gfx11_set_db_render_control(const struct radv_device *device, unsigned num_samples,
                                      unsigned *db_render_control);
/**
 * Attachment state when recording a renderpass instance.
 *
 * The clear value is valid only if there exists a pending clear.
 */
struct radv_attachment {
   VkFormat format;
   struct radv_image_view *iview;
   VkImageLayout layout;
   VkImageLayout stencil_layout;

   union {
      struct radv_color_buffer_info cb;
      struct radv_ds_buffer_info ds;
   };

   struct radv_image_view *resolve_iview;
   VkResolveModeFlagBits resolve_mode;
   VkResolveModeFlagBits stencil_resolve_mode;
   VkImageLayout resolve_layout;
   VkImageLayout stencil_resolve_layout;
};

struct radv_rendering_state {
   bool active;
   bool has_image_views;
   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;
   uint32_t color_samples;
   uint32_t ds_samples;
   uint32_t max_samples;
   struct radv_sample_locations_state sample_locations;
   uint32_t color_att_count;
   struct radv_attachment color_att[MAX_RTS];
   struct radv_attachment ds_att;
   VkImageAspectFlags ds_att_aspects;
   struct radv_attachment vrs_att;
   VkExtent2D vrs_texel_size;
};

struct radv_descriptor_state {
   struct radv_descriptor_set *sets[MAX_SETS];
   uint32_t dirty;
   uint32_t valid;
   struct radv_push_descriptor_set push_set;
   uint32_t dynamic_buffers[4 * MAX_DYNAMIC_BUFFERS];
   uint64_t descriptor_buffers[MAX_SETS];
   bool need_indirect_descriptor_sets;
};

struct radv_push_constant_state {
   uint32_t size;
   uint32_t dynamic_offset_count;
};

enum rgp_flush_bits {
   RGP_FLUSH_WAIT_ON_EOP_TS = 0x1,
   RGP_FLUSH_VS_PARTIAL_FLUSH = 0x2,
   RGP_FLUSH_PS_PARTIAL_FLUSH = 0x4,
   RGP_FLUSH_CS_PARTIAL_FLUSH = 0x8,
   RGP_FLUSH_PFP_SYNC_ME = 0x10,
   RGP_FLUSH_SYNC_CP_DMA = 0x20,
   RGP_FLUSH_INVAL_VMEM_L0 = 0x40,
   RGP_FLUSH_INVAL_ICACHE = 0x80,
   RGP_FLUSH_INVAL_SMEM_L0 = 0x100,
   RGP_FLUSH_FLUSH_L2 = 0x200,
   RGP_FLUSH_INVAL_L2 = 0x400,
   RGP_FLUSH_FLUSH_CB = 0x800,
   RGP_FLUSH_INVAL_CB = 0x1000,
   RGP_FLUSH_FLUSH_DB = 0x2000,
   RGP_FLUSH_INVAL_DB = 0x4000,
   RGP_FLUSH_INVAL_L1 = 0x8000,
};

struct radv_multisample_state {
   bool sample_shading_enable;
   float min_sample_shading;
};

struct radv_ia_multi_vgt_param_helpers {
   uint32_t base;
   bool partial_es_wave;
   bool ia_switch_on_eoi;
   bool partial_vs_wave;
};

struct radv_cmd_state {
   /* Vertex descriptors */
   uint64_t vb_va;
   unsigned vb_size;

   bool predicating;
   uint64_t dirty;

   VkShaderStageFlags active_stages;
   struct radv_shader *shaders[MESA_VULKAN_SHADER_STAGES];
   struct radv_shader *gs_copy_shader;
   struct radv_shader *last_vgt_shader;
   struct radv_shader *rt_prolog;

   struct radv_shader_object *shader_objs[MESA_VULKAN_SHADER_STAGES];

   uint32_t prefetch_L2_mask;

   struct radv_graphics_pipeline *graphics_pipeline;
   struct radv_graphics_pipeline *emitted_graphics_pipeline;
   struct radv_compute_pipeline *compute_pipeline;
   struct radv_compute_pipeline *emitted_compute_pipeline;
   struct radv_ray_tracing_pipeline *rt_pipeline; /* emitted = emitted_compute_pipeline */
   struct radv_dynamic_state dynamic;
   struct radv_vs_input_state dynamic_vs_input;
   struct radv_streamout_state streamout;

   struct radv_rendering_state render;

   /* Index buffer */
   uint32_t index_type;
   uint32_t max_index_count;
   uint64_t index_va;
   int32_t last_index_type;

   uint32_t last_primitive_reset_index; /* only relevant on GFX6-7 */
   enum radv_cmd_flush_bits flush_bits;
   unsigned active_occlusion_queries;
   bool perfect_occlusion_queries_enabled;
   unsigned active_pipeline_queries;
   unsigned active_pipeline_gds_queries;
   unsigned active_pipeline_ace_queries; /* Task shader invocations query */
   unsigned active_prims_gen_queries;
   unsigned active_prims_xfb_queries;
   unsigned active_prims_gen_gds_queries;
   unsigned active_prims_xfb_gds_queries;
   uint32_t trace_id;
   uint32_t last_ia_multi_vgt_param;
   uint32_t last_ge_cntl;

   uint32_t last_num_instances;
   uint32_t last_first_instance;
   bool last_vertex_offset_valid;
   uint32_t last_vertex_offset;
   uint32_t last_drawid;
   uint32_t last_subpass_color_count;

   uint32_t last_sx_ps_downconvert;
   uint32_t last_sx_blend_opt_epsilon;
   uint32_t last_sx_blend_opt_control;

   uint32_t last_db_count_control;

   uint32_t last_db_shader_control;

   /* Whether CP DMA is busy/idle. */
   bool dma_is_busy;

   /* Whether any images that are not L2 coherent are dirty from the CB. */
   bool rb_noncoherent_dirty;

   /* Conditional rendering info. */
   uint8_t predication_op; /* 32-bit or 64-bit predicate value */
   int predication_type;   /* -1: disabled, 0: normal, 1: inverted */
   uint64_t predication_va;
   uint64_t mec_inv_pred_va;  /* For inverted predication when using MEC. */
   bool mec_inv_pred_emitted; /* To ensure we don't have to repeat inverting the VA. */

   /* Inheritance info. */
   VkQueryPipelineStatisticFlags inherited_pipeline_statistics;
   bool inherited_occlusion_queries;
   VkQueryControlFlags inherited_query_control_flags;

   bool context_roll_without_scissor_emitted;

   /* SQTT related state. */
   uint32_t current_event_type;
   uint32_t num_events;
   uint32_t num_layout_transitions;
   bool in_barrier;
   bool pending_sqtt_barrier_end;
   enum rgp_flush_bits sqtt_flush_bits;

   /* NGG culling state. */
   bool has_nggc;

   /* Mesh shading state. */
   bool mesh_shading;

   uint8_t cb_mip[MAX_RTS];
   uint8_t ds_mip;

   /* Whether DRAW_{INDEX}_INDIRECT_{MULTI} is emitted. */
   bool uses_draw_indirect;

   uint32_t rt_stack_size;

   struct radv_shader_part *emitted_vs_prolog;
   uint32_t vbo_misaligned_mask;
   uint32_t vbo_misaligned_mask_invalid;
   uint32_t vbo_bound_mask;

   struct radv_shader_part *emitted_ps_epilog;

   /* Per-vertex VRS state. */
   uint32_t last_vrs_rates;
   int8_t last_vrs_rates_sgpr_idx;

   /* Whether to suspend streamout for internal driver operations. */
   bool suspend_streamout;

   /* Whether this commandbuffer uses performance counters. */
   bool uses_perf_counters;

   struct radv_ia_multi_vgt_param_helpers ia_multi_vgt_param;

   /* Tessellation info when patch control points is dynamic. */
   unsigned tess_num_patches;
   unsigned tess_lds_size;

   unsigned col_format_non_compacted;

   /* Binning state */
   unsigned last_pa_sc_binner_cntl_0;

   struct radv_multisample_state ms;

   /* Custom blend mode for internal operations. */
   unsigned custom_blend_mode;
   unsigned db_render_control;

   unsigned rast_prim;

   uint32_t vtx_base_sgpr;
   uint8_t vtx_emit_num;
   bool uses_drawid;
   bool uses_baseinstance;

   bool uses_out_of_order_rast;
   bool uses_vrs_attachment;
   bool uses_dynamic_patch_control_points;
   bool uses_dynamic_vertex_binding_stride;
};

struct radv_cmd_buffer_upload {
   uint8_t *map;
   unsigned offset;
   uint64_t size;
   struct radeon_winsys_bo *upload_bo;
   struct list_head list;
};

struct radv_cmd_buffer {
   struct vk_command_buffer vk;

   VkCommandBufferUsageFlags usage_flags;
   struct radeon_cmdbuf *cs;
   struct radv_cmd_state state;
   struct radv_buffer *vertex_binding_buffers[MAX_VBS];
   struct radv_vertex_binding vertex_bindings[MAX_VBS];
   uint32_t used_vertex_bindings;
   struct radv_streamout_binding streamout_bindings[MAX_SO_BUFFERS];
   enum radv_queue_family qf;

   uint8_t push_constants[MAX_PUSH_CONSTANTS_SIZE];
   VkShaderStageFlags push_constant_stages;
   struct radv_descriptor_set_header meta_push_descriptors;

   struct radv_descriptor_state descriptors[MAX_BIND_POINTS];

   struct radv_push_constant_state push_constant_state[MAX_BIND_POINTS];

   uint64_t descriptor_buffers[MAX_SETS];

   struct radv_cmd_buffer_upload upload;

   uint32_t scratch_size_per_wave_needed;
   uint32_t scratch_waves_wanted;
   uint32_t compute_scratch_size_per_wave_needed;
   uint32_t compute_scratch_waves_wanted;
   uint32_t esgs_ring_size_needed;
   uint32_t gsvs_ring_size_needed;
   bool tess_rings_needed;
   bool task_rings_needed;
   bool mesh_scratch_ring_needed;
   bool gds_needed;    /* for GFX10 streamout and NGG GS queries */
   bool gds_oa_needed; /* for GFX10 streamout */
   bool sample_positions_needed;
   bool has_indirect_pipeline_binds;

   uint64_t gfx9_fence_va;
   uint32_t gfx9_fence_idx;
   uint64_t gfx9_eop_bug_va;

   struct set vs_prologs;
   struct set ps_epilogs;

   /**
    * Gang state.
    * Used when the command buffer needs work done on a different queue
    * (eg. when a graphics command buffer needs compute work).
    * Currently only one follower is possible per command buffer.
    */
   struct {
      /** Follower command stream. */
      struct radeon_cmdbuf *cs;

      /** Flush bits for the follower cmdbuf. */
      enum radv_cmd_flush_bits flush_bits;

      /**
       * For synchronization between the follower and leader.
       * The value of these semaphores are incremented whenever we
       * encounter a barrier that affects the follower.
       *
       * DWORD 0: Leader to follower semaphore.
       *          The leader writes the value and the follower waits.
       * DWORD 1: Follower to leader semaphore.
       *          The follower writes the value, and the leader waits.
       */
      struct {
         uint64_t va;                     /* Virtual address of the semaphore. */
         uint32_t leader_value;           /* Current value of the leader. */
         uint32_t emitted_leader_value;   /* Last value emitted by the leader. */
         uint32_t follower_value;         /* Current value of the follower. */
         uint32_t emitted_follower_value; /* Last value emitted by the follower. */
      } sem;
   } gang;

   /**
    * Whether a query pool has been reset and we have to flush caches.
    */
   bool pending_reset_query;

   /**
    * Bitmask of pending active query flushes.
    */
   enum radv_cmd_flush_bits active_query_flush_bits;

   struct {
      struct radv_video_session *vid;
      struct radv_video_session_params *params;
      struct rvcn_sq_var sq;
      struct rvcn_decode_buffer_s *decode_buffer;
   } video;

   struct {
      /* Temporary space for some transfer queue copy command workarounds. */
      struct radeon_winsys_bo *copy_temp;
   } transfer;

   uint64_t shader_upload_seq;

   uint32_t sqtt_cb_id;

   struct util_dynarray ray_history;
};

static inline struct radv_device *
radv_cmd_buffer_device(const struct radv_cmd_buffer *cmd_buffer)
{
   return (struct radv_device *)cmd_buffer->vk.base.device;
}

static inline bool
radv_cmdbuf_has_stage(const struct radv_cmd_buffer *cmd_buffer, gl_shader_stage stage)
{
   return !!(cmd_buffer->state.active_stages & mesa_to_vk_shader_stage(stage));
}

static inline uint32_t
radv_get_num_pipeline_stat_queries(struct radv_cmd_buffer *cmd_buffer)
{
   /* SAMPLE_STREAMOUTSTATS also requires PIPELINESTAT_START to be enabled. */
   return cmd_buffer->state.active_pipeline_queries + cmd_buffer->state.active_prims_gen_queries +
          cmd_buffer->state.active_prims_xfb_queries;
}

extern const struct vk_command_buffer_ops radv_cmd_buffer_ops;

struct radv_dispatch_info {
   /**
    * Determine the layout of the grid (in block units) to be used.
    */
   uint32_t blocks[3];

   /**
    * A starting offset for the grid. If unaligned is set, the offset
    * must still be aligned.
    */
   uint32_t offsets[3];

   /**
    * Whether it's an unaligned compute dispatch.
    */
   bool unaligned;

   /**
    * Whether waves must be launched in order.
    */
   bool ordered;

   /**
    * Indirect compute parameters resource.
    */
   struct radeon_winsys_bo *indirect;
   uint64_t va;
};

void radv_compute_dispatch(struct radv_cmd_buffer *cmd_buffer, const struct radv_dispatch_info *info);

bool radv_cmd_buffer_uses_mec(struct radv_cmd_buffer *cmd_buffer);

void radv_emit_graphics(struct radv_device *device, struct radeon_cmdbuf *cs);
void radv_emit_compute(struct radv_device *device, struct radeon_cmdbuf *cs);

void radv_create_gfx_config(struct radv_device *device);

void radv_write_scissors(struct radeon_cmdbuf *cs, int count, const VkRect2D *scissors, const VkViewport *viewports);

void radv_write_guardband(struct radeon_cmdbuf *cs, int count, const VkViewport *viewports, unsigned rast_prim,
                          unsigned polygon_mode, float line_width);

VkResult radv_create_shadow_regs_preamble(struct radv_device *device, struct radv_queue_state *queue_state);
void radv_destroy_shadow_regs_preamble(struct radv_device *device, struct radv_queue_state *queue_state,
                                       struct radeon_winsys *ws);
void radv_emit_shadow_regs_preamble(struct radeon_cmdbuf *cs, const struct radv_device *device,
                                    struct radv_queue_state *queue_state);
VkResult radv_init_shadowed_regs_buffer_state(const struct radv_device *device, struct radv_queue *queue);

uint32_t radv_get_ia_multi_vgt_param(struct radv_cmd_buffer *cmd_buffer, bool instanced_draw, bool indirect_draw,
                                     bool count_from_stream_output, uint32_t draw_vertex_count, unsigned topology,
                                     bool prim_restart_enable, unsigned patch_control_points,
                                     unsigned num_tess_patches);
void radv_cs_emit_write_event_eop(struct radeon_cmdbuf *cs, enum amd_gfx_level gfx_level, enum radv_queue_family qf,
                                  unsigned event, unsigned event_flags, unsigned dst_sel, unsigned data_sel,
                                  uint64_t va, uint32_t new_fence, uint64_t gfx9_eop_bug_va);

struct radv_vgt_shader_key {
   uint8_t tess : 1;
   uint8_t gs : 1;
   uint8_t mesh_scratch_ring : 1;
   uint8_t mesh : 1;
   uint8_t ngg_passthrough : 1;
   uint8_t ngg : 1; /* gfx10+ */
   uint8_t ngg_streamout : 1;
   uint8_t hs_wave32 : 1;
   uint8_t gs_wave32 : 1;
   uint8_t vs_wave32 : 1;
};

void radv_cs_emit_cache_flush(struct radeon_winsys *ws, struct radeon_cmdbuf *cs, enum amd_gfx_level gfx_level,
                              uint32_t *flush_cnt, uint64_t flush_va, enum radv_queue_family qf,
                              enum radv_cmd_flush_bits flush_bits, enum rgp_flush_bits *sqtt_flush_bits,
                              uint64_t gfx9_eop_bug_va);
void radv_emit_cache_flush(struct radv_cmd_buffer *cmd_buffer);
void radv_emit_set_predication_state(struct radv_cmd_buffer *cmd_buffer, bool draw_visible, unsigned pred_op,
                                     uint64_t va);
void radv_emit_cond_exec(const struct radv_device *device, struct radeon_cmdbuf *cs, uint64_t va, uint32_t count);

void radv_cp_dma_buffer_copy(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dest_va, uint64_t size);
void radv_cs_cp_dma_prefetch(const struct radv_device *device, struct radeon_cmdbuf *cs, uint64_t va, unsigned size,
                             bool predicating);
void radv_cp_dma_prefetch(struct radv_cmd_buffer *cmd_buffer, uint64_t va, unsigned size);
void radv_cp_dma_clear_buffer(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, unsigned value);
void radv_cp_dma_wait_for_idle(struct radv_cmd_buffer *cmd_buffer);

uint32_t radv_get_vgt_index_size(uint32_t type);

void radv_emit_vgt_shader_config(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs,
                                 const struct radv_vgt_shader_key *key);

void radv_emit_blend_state(struct radeon_cmdbuf *ctx_cs, const struct radv_shader *ps, uint32_t spi_shader_col_format,
                           uint32_t cb_shader_mask);

unsigned radv_instance_rate_prolog_index(unsigned num_attributes, uint32_t instance_rate_inputs);

struct radv_ps_epilog_state {
   uint8_t color_attachment_count;
   VkFormat color_attachment_formats[MAX_RTS];

   uint32_t color_write_mask;
   uint32_t color_blend_enable;

   uint32_t colors_written;
   bool mrt0_is_dual_src;
   bool export_depth;
   bool export_stencil;
   bool export_sample_mask;
   bool alpha_to_coverage_via_mrtz;
   bool alpha_to_one;
   uint8_t need_src_alpha;
};

struct radv_ps_epilog_key radv_generate_ps_epilog_key(const struct radv_device *device,
                                                      const struct radv_ps_epilog_state *state);

bool radv_needs_null_export_workaround(const struct radv_device *device, const struct radv_shader *ps,
                                       unsigned custom_blend_mode);

void radv_cmd_buffer_reset_rendering(struct radv_cmd_buffer *cmd_buffer);
bool radv_cmd_buffer_upload_alloc_aligned(struct radv_cmd_buffer *cmd_buffer, unsigned size, unsigned alignment,
                                          unsigned *out_offset, void **ptr);
bool radv_cmd_buffer_upload_alloc(struct radv_cmd_buffer *cmd_buffer, unsigned size, unsigned *out_offset, void **ptr);
bool radv_cmd_buffer_upload_data(struct radv_cmd_buffer *cmd_buffer, unsigned size, const void *data,
                                 unsigned *out_offset);
void radv_write_vertex_descriptors(const struct radv_cmd_buffer *cmd_buffer,
                                   const struct radv_graphics_pipeline *pipeline, bool full_null_descriptors,
                                   void *vb_ptr);

void radv_emit_default_sample_locations(struct radeon_cmdbuf *cs, int nr_samples);
unsigned radv_get_default_max_sample_dist(int log_samples);
void radv_device_init_msaa(struct radv_device *device);
VkResult radv_device_init_vrs_state(struct radv_device *device);

void radv_cs_write_data_imm(struct radeon_cmdbuf *cs, unsigned engine_sel, uint64_t va, uint32_t imm);

void radv_update_ds_clear_metadata(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                                   VkClearDepthStencilValue ds_clear_value, VkImageAspectFlags aspects);

void radv_update_color_clear_metadata(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                                      int cb_idx, uint32_t color_values[2]);

void radv_update_fce_metadata(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                              const VkImageSubresourceRange *range, bool value);

void radv_update_dcc_metadata(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                              const VkImageSubresourceRange *range, bool value);
enum radv_cmd_flush_bits radv_src_access_flush(struct radv_cmd_buffer *cmd_buffer, VkAccessFlags2 src_flags,
                                               const struct radv_image *image);
enum radv_cmd_flush_bits radv_dst_access_flush(struct radv_cmd_buffer *cmd_buffer, VkAccessFlags2 dst_flags,
                                               const struct radv_image *image);

void radv_cmd_buffer_trace_emit(struct radv_cmd_buffer *cmd_buffer);

void radv_cmd_buffer_annotate(struct radv_cmd_buffer *cmd_buffer, const char *annotation);

bool radv_get_memory_fd(struct radv_device *device, struct radv_device_memory *memory, int *pFD);
void radv_free_memory(struct radv_device *device, const VkAllocationCallbacks *pAllocator,
                      struct radv_device_memory *mem);

static inline void
radv_emit_shader_pointer_head(struct radeon_cmdbuf *cs, unsigned sh_offset, unsigned pointer_count,
                              bool use_32bit_pointers)
{
   radeon_emit(cs, PKT3(PKT3_SET_SH_REG, pointer_count * (use_32bit_pointers ? 1 : 2), 0));
   radeon_emit(cs, (sh_offset - SI_SH_REG_OFFSET) >> 2);
}

static inline void
radv_emit_shader_pointer_body(const struct radv_device *device, struct radeon_cmdbuf *cs, uint64_t va,
                              bool use_32bit_pointers)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   radeon_emit(cs, va);

   if (use_32bit_pointers) {
      assert(va == 0 || (va >> 32) == pdev->info.address32_hi);
   } else {
      radeon_emit(cs, va >> 32);
   }
}

static inline void
radv_emit_shader_pointer(const struct radv_device *device, struct radeon_cmdbuf *cs, uint32_t sh_offset, uint64_t va,
                         bool global)
{
   bool use_32bit_pointers = !global;

   radv_emit_shader_pointer_head(cs, sh_offset, 1, use_32bit_pointers);
   radv_emit_shader_pointer_body(device, cs, va, use_32bit_pointers);
}

static inline unsigned
vk_to_bind_point(VkPipelineBindPoint bind_point)
{
   return bind_point == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR ? 2 : bind_point;
}

static inline struct radv_descriptor_state *
radv_get_descriptors_state(struct radv_cmd_buffer *cmd_buffer, VkPipelineBindPoint bind_point)
{
   return &cmd_buffer->descriptors[vk_to_bind_point(bind_point)];
}

static inline const struct radv_push_constant_state *
radv_get_push_constants_state(const struct radv_cmd_buffer *cmd_buffer, VkPipelineBindPoint bind_point)
{
   return &cmd_buffer->push_constant_state[vk_to_bind_point(bind_point)];
}

void radv_get_viewport_xform(const VkViewport *viewport, float scale[3], float translate[3]);

/*
 * Takes x,y,z as exact numbers of invocations, instead of blocks.
 *
 * Limitations: Can't call normal dispatch functions without binding or rebinding
 *              the compute pipeline.
 */
void radv_unaligned_dispatch(struct radv_cmd_buffer *cmd_buffer, uint32_t x, uint32_t y, uint32_t z);

void radv_indirect_dispatch(struct radv_cmd_buffer *cmd_buffer, struct radeon_winsys_bo *bo, uint64_t va);

struct radv_ray_tracing_group;

void radv_pipeline_stage_init(const VkPipelineShaderStageCreateInfo *sinfo, const struct radv_pipeline_layout *layout,
                              const struct radv_shader_stage_key *stage_key, struct radv_shader_stage *out_stage);

void radv_hash_graphics_spirv_to_nir(blake3_hash hash, const struct radv_shader_stage *stage,
                                     const struct radv_spirv_to_nir_options *options);

void radv_hash_shaders(const struct radv_device *device, unsigned char *hash, const struct radv_shader_stage *stages,
                       uint32_t stage_count, const struct radv_pipeline_layout *layout,
                       const struct radv_graphics_state_key *gfx_state);

struct radv_ray_tracing_stage;
void radv_hash_rt_shaders(const struct radv_device *device, unsigned char *hash,
                          const struct radv_ray_tracing_stage *stages,
                          const VkRayTracingPipelineCreateInfoKHR *pCreateInfo,
                          const struct radv_ray_tracing_group *groups);

bool radv_enable_rt(const struct radv_physical_device *pdev, bool rt_pipelines);

bool radv_emulate_rt(const struct radv_physical_device *pdev);

struct radv_prim_vertex_count {
   uint8_t min;
   uint8_t incr;
};

enum radv_pipeline_type {
   RADV_PIPELINE_GRAPHICS,
   RADV_PIPELINE_GRAPHICS_LIB,
   /* Compute pipeline */
   RADV_PIPELINE_COMPUTE,
   /* Raytracing pipeline */
   RADV_PIPELINE_RAY_TRACING,
};

struct radv_pipeline_group_handle {
   uint64_t recursive_shader_ptr;

   union {
      uint32_t general_index;
      uint32_t closest_hit_index;
   };
   union {
      uint32_t intersection_index;
      uint32_t any_hit_index;
   };
};

struct radv_rt_capture_replay_handle {
   struct radv_serialized_shader_arena_block recursive_shader_alloc;
   uint32_t non_recursive_idx;
};

struct radv_pipeline {
   struct vk_object_base base;
   enum radv_pipeline_type type;

   VkPipelineCreateFlags2KHR create_flags;

   struct vk_pipeline_cache_object *cache_object;

   bool is_internal;
   bool need_indirect_descriptor_sets;
   struct radv_shader *shaders[MESA_VULKAN_SHADER_STAGES];
   struct radv_shader *gs_copy_shader;

   struct radeon_cmdbuf cs;
   uint32_t ctx_cs_hash;
   struct radeon_cmdbuf ctx_cs;

   uint32_t user_data_0[MESA_VULKAN_SHADER_STAGES];

   /* Unique pipeline hash identifier. */
   uint64_t pipeline_hash;

   /* Pipeline layout info. */
   uint32_t push_constant_size;
   uint32_t dynamic_offset_count;
};

struct radv_sqtt_shaders_reloc {
   struct radeon_winsys_bo *bo;
   union radv_shader_arena_block *alloc;
   uint64_t va[MESA_VULKAN_SHADER_STAGES];
};

struct radv_graphics_pipeline {
   struct radv_pipeline base;

   bool uses_drawid;
   bool uses_baseinstance;

   /* Whether the pipeline forces per-vertex VRS (GFX10.3+). */
   bool force_vrs_per_vertex;

   /* Whether the pipeline uses NGG (GFX10+). */
   bool is_ngg;
   bool has_ngg_culling;

   uint8_t vtx_emit_num;

   uint32_t vtx_base_sgpr;
   uint64_t dynamic_states;
   uint64_t needed_dynamic_state;

   VkShaderStageFlags active_stages;

   /* Used for rbplus */
   uint32_t col_format_non_compacted;

   struct radv_dynamic_state dynamic_state;

   struct radv_vs_input_state vs_input_state;

   struct radv_multisample_state ms;
   struct radv_ia_multi_vgt_param_helpers ia_multi_vgt_param;
   uint32_t binding_stride[MAX_VBS];
   uint8_t attrib_bindings[MAX_VERTEX_ATTRIBS];
   uint32_t attrib_ends[MAX_VERTEX_ATTRIBS];
   uint32_t attrib_index_offset[MAX_VERTEX_ATTRIBS];
   uint32_t db_render_control;

   /* Last pre-PS API stage */
   gl_shader_stage last_vgt_api_stage;

   unsigned rast_prim;

   /* For vk_graphics_pipeline_state */
   void *state_data;

   /* Custom blend mode for internal operations. */
   unsigned custom_blend_mode;

   /* Whether the pipeline uses out-of-order rasterization. */
   bool uses_out_of_order_rast;

   /* Whether the pipeline uses a VRS attachment. */
   bool uses_vrs_attachment;

   /* For graphics pipeline library */
   bool retain_shaders;

   /* For relocation of shaders with RGP. */
   struct radv_sqtt_shaders_reloc *sqtt_shaders_reloc;
};

struct radv_compute_pipeline {
   struct radv_pipeline base;

   struct {
      uint64_t va;
      uint64_t size;
   } indirect;
};

struct radv_ray_tracing_group {
   VkRayTracingShaderGroupTypeKHR type;
   uint32_t recursive_shader; /* generalShader or closestHitShader */
   uint32_t any_hit_shader;
   uint32_t intersection_shader;
   struct radv_pipeline_group_handle handle;
};

enum radv_rt_const_arg_state {
   RADV_RT_CONST_ARG_STATE_UNINITIALIZED,
   RADV_RT_CONST_ARG_STATE_VALID,
   RADV_RT_CONST_ARG_STATE_INVALID,
};

struct radv_rt_const_arg_info {
   enum radv_rt_const_arg_state state;
   uint32_t value;
};

struct radv_ray_tracing_stage_info {
   bool can_inline;

   BITSET_DECLARE(unused_args, AC_MAX_ARGS);

   struct radv_rt_const_arg_info tmin;
   struct radv_rt_const_arg_info tmax;

   struct radv_rt_const_arg_info sbt_offset;
   struct radv_rt_const_arg_info sbt_stride;

   struct radv_rt_const_arg_info miss_index;

   uint32_t set_flags;
   uint32_t unset_flags;
};

struct radv_ray_tracing_stage {
   struct vk_pipeline_cache_object *nir;
   struct radv_shader *shader;
   gl_shader_stage stage;
   uint32_t stack_size;

   struct radv_ray_tracing_stage_info info;

   uint8_t sha1[SHA1_DIGEST_LENGTH];
};

struct radv_ray_tracing_pipeline {
   struct radv_compute_pipeline base;

   struct radv_shader *prolog;

   struct radv_ray_tracing_stage *stages;
   struct radv_ray_tracing_group *groups;
   unsigned stage_count;
   unsigned non_imported_stage_count;
   unsigned group_count;

   uint8_t sha1[SHA1_DIGEST_LENGTH];
   uint32_t stack_size;

   /* set if any shaders from this pipeline require robustness2 in the merged traversal shader */
   bool traversal_storage_robustness2 : 1;
   bool traversal_uniform_robustness2 : 1;
};

struct radv_retained_shaders {
   struct {
      void *serialized_nir;
      size_t serialized_nir_size;
      unsigned char shader_sha1[SHA1_DIGEST_LENGTH];
      struct radv_shader_stage_key key;
   } stages[MESA_VULKAN_SHADER_STAGES];
};

struct radv_graphics_lib_pipeline {
   struct radv_graphics_pipeline base;

   struct radv_pipeline_layout layout;

   struct vk_graphics_pipeline_state graphics_state;

   VkGraphicsPipelineLibraryFlagsEXT lib_flags;

   struct radv_retained_shaders retained_shaders;

   void *mem_ctx;

   unsigned stage_count;
   VkPipelineShaderStageCreateInfo *stages;
   struct radv_shader_stage_key stage_keys[MESA_VULKAN_SHADER_STAGES];
};

#define RADV_DECL_PIPELINE_DOWNCAST(pipe_type, pipe_enum)                                                              \
   static inline struct radv_##pipe_type##_pipeline *radv_pipeline_to_##pipe_type(struct radv_pipeline *pipeline)      \
   {                                                                                                                   \
      assert(pipeline->type == pipe_enum);                                                                             \
      return (struct radv_##pipe_type##_pipeline *)pipeline;                                                           \
   }

RADV_DECL_PIPELINE_DOWNCAST(graphics, RADV_PIPELINE_GRAPHICS)
RADV_DECL_PIPELINE_DOWNCAST(graphics_lib, RADV_PIPELINE_GRAPHICS_LIB)
RADV_DECL_PIPELINE_DOWNCAST(compute, RADV_PIPELINE_COMPUTE)
RADV_DECL_PIPELINE_DOWNCAST(ray_tracing, RADV_PIPELINE_RAY_TRACING)

struct radv_shader_layout {
   uint32_t num_sets;

   struct {
      struct radv_descriptor_set_layout *layout;
      uint32_t dynamic_offset_start;
   } set[MAX_SETS];

   uint32_t push_constant_size;
   uint32_t dynamic_offset_count;
   bool use_dynamic_descriptors;
};

struct radv_shader_stage {
   gl_shader_stage stage;
   gl_shader_stage next_stage;

   struct {
      const struct vk_object_base *object;
      const char *data;
      uint32_t size;
   } spirv;

   const char *entrypoint;
   const VkSpecializationInfo *spec_info;

   unsigned char shader_sha1[20];

   nir_shader *nir;
   nir_shader *internal_nir; /* meta shaders */

   struct radv_shader_info info;
   struct radv_shader_args args;
   struct radv_shader_stage_key key;

   VkPipelineCreationFeedback feedback;

   struct radv_shader_layout layout;
};

void radv_shader_layout_init(const struct radv_pipeline_layout *pipeline_layout, gl_shader_stage stage,
                             struct radv_shader_layout *layout);

static inline bool
radv_is_last_vgt_stage(const struct radv_shader_stage *stage)
{
   return (stage->info.stage == MESA_SHADER_VERTEX || stage->info.stage == MESA_SHADER_TESS_EVAL ||
           stage->info.stage == MESA_SHADER_GEOMETRY || stage->info.stage == MESA_SHADER_MESH) &&
          (stage->info.next_stage == MESA_SHADER_FRAGMENT || stage->info.next_stage == MESA_SHADER_NONE);
}

static inline bool
radv_pipeline_has_stage(const struct radv_graphics_pipeline *pipeline, gl_shader_stage stage)
{
   return pipeline->base.shaders[stage];
}

bool radv_pipeline_has_gs_copy_shader(const struct radv_pipeline *pipeline);

const struct radv_userdata_info *radv_get_user_sgpr(const struct radv_shader *shader, int idx);

struct radv_shader *radv_get_shader(struct radv_shader *const *shaders, gl_shader_stage stage);

void radv_emit_compute_shader(const struct radv_physical_device *pdev, struct radeon_cmdbuf *cs,
                              const struct radv_shader *shader);

bool radv_mem_vectorize_callback(unsigned align_mul, unsigned align_offset, unsigned bit_size, unsigned num_components,
                                 nir_intrinsic_instr *low, nir_intrinsic_instr *high, void *data);

void radv_emit_vertex_shader(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs, struct radeon_cmdbuf *cs,
                             const struct radv_shader *vs, const struct radv_shader *next_stage);

void radv_emit_tess_ctrl_shader(const struct radv_device *device, struct radeon_cmdbuf *cs,
                                const struct radv_shader *tcs);

void radv_emit_tess_eval_shader(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs,
                                struct radeon_cmdbuf *cs, const struct radv_shader *tes, const struct radv_shader *gs);

void radv_emit_fragment_shader(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs, struct radeon_cmdbuf *cs,
                               const struct radv_shader *ps);

void radv_emit_ps_inputs(const struct radv_device *device, struct radeon_cmdbuf *cs,
                         const struct radv_shader *last_vgt_shader, const struct radv_shader *ps);

struct radv_ia_multi_vgt_param_helpers radv_compute_ia_multi_vgt_param(const struct radv_device *device,
                                                                       struct radv_shader *const *shaders);

void radv_emit_vgt_reuse(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs, const struct radv_shader *tes,
                         const struct radv_vgt_shader_key *key);

void radv_emit_vgt_gs_out(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs,
                          uint32_t vgt_gs_out_prim_type);

void radv_emit_vgt_gs_mode(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs,
                           const struct radv_shader *last_vgt_api_shader);

void gfx103_emit_vgt_draw_payload_cntl(struct radeon_cmdbuf *ctx_cs, const struct radv_shader *mesh_shader,
                                       bool enable_vrs);

void gfx103_emit_vrs_state(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs, const struct radv_shader *ps,
                           bool enable_vrs, bool enable_vrs_coarse_shading, bool force_vrs_per_vertex);

void radv_emit_geometry_shader(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs, struct radeon_cmdbuf *cs,
                               const struct radv_shader *gs, const struct radv_shader *es,
                               const struct radv_shader *gs_copy_shader);

void radv_emit_mesh_shader(const struct radv_device *device, struct radeon_cmdbuf *ctx_cs, struct radeon_cmdbuf *cs,
                           const struct radv_shader *ms);

void radv_graphics_shaders_compile(struct radv_device *device, struct vk_pipeline_cache *cache,
                                   struct radv_shader_stage *stages, const struct radv_graphics_state_key *gfx_state,
                                   bool keep_executable_info, bool keep_statistic_info, bool is_internal,
                                   struct radv_retained_shaders *retained_shaders, bool noop_fs,
                                   struct radv_shader **shaders, struct radv_shader_binary **binaries,
                                   struct radv_shader **gs_copy_shader, struct radv_shader_binary **gs_copy_binary);

void radv_compute_pipeline_init(const struct radv_device *device, struct radv_compute_pipeline *pipeline,
                                const struct radv_pipeline_layout *layout, struct radv_shader *shader);

struct radv_shader *radv_compile_cs(struct radv_device *device, struct vk_pipeline_cache *cache,
                                    struct radv_shader_stage *cs_stage, bool keep_executable_info,
                                    bool keep_statistic_info, bool is_internal, struct radv_shader_binary **cs_binary);

struct radv_graphics_pipeline_create_info {
   bool use_rectlist;
   bool db_depth_clear;
   bool db_stencil_clear;
   bool depth_compress_disable;
   bool stencil_compress_disable;
   bool resummarize_enable;
   uint32_t custom_blend_mode;
};

struct radv_shader_stage_key radv_pipeline_get_shader_key(const struct radv_device *device,
                                                          const VkPipelineShaderStageCreateInfo *stage,
                                                          VkPipelineCreateFlags2KHR flags, const void *pNext);

void radv_pipeline_init(struct radv_device *device, struct radv_pipeline *pipeline, enum radv_pipeline_type type);

VkResult radv_graphics_pipeline_create(VkDevice device, VkPipelineCache cache,
                                       const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                       const struct radv_graphics_pipeline_create_info *extra,
                                       const VkAllocationCallbacks *alloc, VkPipeline *pPipeline);

VkResult radv_compute_pipeline_create(VkDevice _device, VkPipelineCache _cache,
                                      const VkComputePipelineCreateInfo *pCreateInfo,
                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline);

bool radv_pipeline_capture_shaders(const struct radv_device *device, VkPipelineCreateFlags2KHR flags);
bool radv_pipeline_capture_shader_stats(const struct radv_device *device, VkPipelineCreateFlags2KHR flags);

VkPipelineShaderStageCreateInfo *radv_copy_shader_stage_create_info(struct radv_device *device, uint32_t stageCount,
                                                                    const VkPipelineShaderStageCreateInfo *pStages,
                                                                    void *mem_ctx);

bool radv_shader_need_indirect_descriptor_sets(const struct radv_shader *shader);

bool radv_pipeline_has_ngg(const struct radv_graphics_pipeline *pipeline);

void radv_pipeline_destroy(struct radv_device *device, struct radv_pipeline *pipeline,
                           const VkAllocationCallbacks *allocator);

struct vk_format_description;
bool radv_device_supports_etc(const struct radv_physical_device *pdev);

unsigned radv_get_dcc_max_uncompressed_block_size(const struct radv_device *device, const struct radv_image *image);

VkResult radv_image_from_gralloc(VkDevice device_h, const VkImageCreateInfo *base_info,
                                 const VkNativeBufferANDROID *gralloc_info, const VkAllocationCallbacks *alloc,
                                 VkImage *out_image_h);
VkResult radv_import_ahb_memory(struct radv_device *device, struct radv_device_memory *mem, unsigned priority,
                                const VkImportAndroidHardwareBufferInfoANDROID *info);
VkResult radv_create_ahb_memory(struct radv_device *device, struct radv_device_memory *mem, unsigned priority,
                                const VkMemoryAllocateInfo *pAllocateInfo);

unsigned radv_ahb_format_for_vk_format(VkFormat vk_format);

VkFormat radv_select_android_external_format(const void *next, VkFormat default_format);

bool radv_android_gralloc_supports_format(VkFormat format, VkImageUsageFlagBits usage);

struct radv_resolve_barrier {
   VkPipelineStageFlags2 src_stage_mask;
   VkPipelineStageFlags2 dst_stage_mask;
   VkAccessFlags2 src_access_mask;
   VkAccessFlags2 dst_access_mask;
};

void radv_emit_resolve_barrier(struct radv_cmd_buffer *cmd_buffer, const struct radv_resolve_barrier *barrier);

bool radv_queue_internal_submit(struct radv_queue *queue, struct radeon_cmdbuf *cs);

int radv_queue_init(struct radv_device *device, struct radv_queue *queue, int idx,
                    const VkDeviceQueueCreateInfo *create_info,
                    const VkDeviceQueueGlobalPriorityCreateInfoKHR *global_priority);

void radv_set_descriptor_set(struct radv_cmd_buffer *cmd_buffer, VkPipelineBindPoint bind_point,
                             struct radv_descriptor_set *set, unsigned idx);

void radv_meta_push_descriptor_set(struct radv_cmd_buffer *cmd_buffer, VkPipelineBindPoint pipelineBindPoint,
                                   VkPipelineLayout _layout, uint32_t set, uint32_t descriptorWriteCount,
                                   const VkWriteDescriptorSet *pDescriptorWrites);

uint32_t radv_init_dcc(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                       const VkImageSubresourceRange *range, uint32_t value);

uint32_t radv_init_fmask(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                         const VkImageSubresourceRange *range);

/* radv_nir_to_llvm.c */
struct radv_shader_args;
struct radv_nir_compiler_options;
struct radv_shader_info;

void llvm_compile_shader(const struct radv_nir_compiler_options *options, const struct radv_shader_info *info,
                         unsigned shader_count, struct nir_shader *const *shaders, struct radv_shader_binary **binary,
                         const struct radv_shader_args *args);

bool radv_sqtt_init(struct radv_device *device);
void radv_sqtt_finish(struct radv_device *device);
bool radv_begin_sqtt(struct radv_queue *queue);
bool radv_end_sqtt(struct radv_queue *queue);
bool radv_get_sqtt_trace(struct radv_queue *queue, struct ac_sqtt_trace *sqtt_trace);
void radv_reset_sqtt_trace(struct radv_device *device);
void radv_emit_sqtt_userdata(const struct radv_cmd_buffer *cmd_buffer, const void *data, uint32_t num_dwords);
bool radv_is_instruction_timing_enabled(void);
bool radv_sqtt_queue_events_enabled(void);
bool radv_sqtt_sample_clocks(struct radv_device *device);

void radv_emit_inhibit_clockgating(const struct radv_device *device, struct radeon_cmdbuf *cs, bool inhibit);
void radv_emit_spi_config_cntl(const struct radv_device *device, struct radeon_cmdbuf *cs, bool enable);

VkResult radv_sqtt_get_timed_cmdbuf(struct radv_queue *queue, struct radeon_winsys_bo *timestamp_bo,
                                    uint32_t timestamp_offset, VkPipelineStageFlags2 timestamp_stage,
                                    VkCommandBuffer *pcmdbuf);

VkResult radv_sqtt_acquire_gpu_timestamp(struct radv_device *device, struct radeon_winsys_bo **gpu_timestamp_bo,
                                         uint32_t *gpu_timestamp_offset, void **gpu_timestamp_ptr);

VkResult radv_rra_trace_init(struct radv_device *device);

VkResult radv_rra_dump_trace(VkQueue vk_queue, char *filename);
void radv_rra_trace_clear_ray_history(VkDevice _device, struct radv_rra_trace_data *data);
void radv_rra_trace_finish(VkDevice vk_device, struct radv_rra_trace_data *data);

void radv_memory_trace_init(struct radv_device *device);
void radv_rmv_log_bo_allocate(struct radv_device *device, struct radeon_winsys_bo *bo, bool is_internal);
void radv_rmv_log_bo_destroy(struct radv_device *device, struct radeon_winsys_bo *bo);
void radv_rmv_log_heap_create(struct radv_device *device, VkDeviceMemory heap, bool is_internal,
                              VkMemoryAllocateFlags alloc_flags);
void radv_rmv_log_buffer_bind(struct radv_device *device, VkBuffer _buffer);
void radv_rmv_log_image_create(struct radv_device *device, const VkImageCreateInfo *create_info, bool is_internal,
                               VkImage _image);
void radv_rmv_log_image_bind(struct radv_device *device, VkImage _image);
void radv_rmv_log_query_pool_create(struct radv_device *device, VkQueryPool pool);
void radv_rmv_log_command_buffer_bo_create(struct radv_device *device, struct radeon_winsys_bo *bo,
                                           uint32_t executable_size, uint32_t data_size, uint32_t scratch_size);
void radv_rmv_log_command_buffer_bo_destroy(struct radv_device *device, struct radeon_winsys_bo *bo);
void radv_rmv_log_border_color_palette_create(struct radv_device *device, struct radeon_winsys_bo *bo);
void radv_rmv_log_border_color_palette_destroy(struct radv_device *device, struct radeon_winsys_bo *bo);
void radv_rmv_log_sparse_add_residency(struct radv_device *device, struct radeon_winsys_bo *src_bo, uint64_t offset);
void radv_rmv_log_sparse_remove_residency(struct radv_device *device, struct radeon_winsys_bo *src_bo, uint64_t offset);
void radv_rmv_log_descriptor_pool_create(struct radv_device *device, const VkDescriptorPoolCreateInfo *create_info,
                                         VkDescriptorPool pool);
void radv_rmv_log_graphics_pipeline_create(struct radv_device *device, struct radv_pipeline *pipeline,
                                           bool is_internal);
void radv_rmv_log_compute_pipeline_create(struct radv_device *device, struct radv_pipeline *pipeline, bool is_internal);
void radv_rmv_log_rt_pipeline_create(struct radv_device *device, struct radv_ray_tracing_pipeline *pipeline);
void radv_rmv_log_event_create(struct radv_device *device, VkEvent event, VkEventCreateFlags flags, bool is_internal);
void radv_rmv_log_resource_destroy(struct radv_device *device, uint64_t handle);
void radv_rmv_log_submit(struct radv_device *device, enum amd_ip_type type);
void radv_rmv_fill_device_info(const struct radv_physical_device *pdev, struct vk_rmv_device_info *info);
void radv_rmv_collect_trace_events(struct radv_device *device);
void radv_memory_trace_finish(struct radv_device *device);

VkResult radv_alloc_memory(struct radv_device *device, const VkMemoryAllocateInfo *pAllocateInfo,
                           const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem, bool is_internal);

/* radv_sqtt_layer_.c */
struct radv_barrier_data {
   union {
      struct {
         uint16_t depth_stencil_expand : 1;
         uint16_t htile_hiz_range_expand : 1;
         uint16_t depth_stencil_resummarize : 1;
         uint16_t dcc_decompress : 1;
         uint16_t fmask_decompress : 1;
         uint16_t fast_clear_eliminate : 1;
         uint16_t fmask_color_expand : 1;
         uint16_t init_mask_ram : 1;
         uint16_t reserved : 8;
      };
      uint16_t all;
   } layout_transitions;
};

/**
 * Value for the reason field of an RGP barrier start marker originating from
 * the Vulkan client (does not include PAL-defined values). (Table 15)
 */
enum rgp_barrier_reason {
   RGP_BARRIER_UNKNOWN_REASON = 0xFFFFFFFF,

   /* External app-generated barrier reasons, i.e. API synchronization
    * commands Range of valid values: [0x00000001 ... 0x7FFFFFFF].
    */
   RGP_BARRIER_EXTERNAL_CMD_PIPELINE_BARRIER = 0x00000001,
   RGP_BARRIER_EXTERNAL_RENDER_PASS_SYNC = 0x00000002,
   RGP_BARRIER_EXTERNAL_CMD_WAIT_EVENTS = 0x00000003,

   /* Internal barrier reasons, i.e. implicit synchronization inserted by
    * the Vulkan driver Range of valid values: [0xC0000000 ... 0xFFFFFFFE].
    */
   RGP_BARRIER_INTERNAL_BASE = 0xC0000000,
   RGP_BARRIER_INTERNAL_PRE_RESET_QUERY_POOL_SYNC = RGP_BARRIER_INTERNAL_BASE + 0,
   RGP_BARRIER_INTERNAL_POST_RESET_QUERY_POOL_SYNC = RGP_BARRIER_INTERNAL_BASE + 1,
   RGP_BARRIER_INTERNAL_GPU_EVENT_RECYCLE_STALL = RGP_BARRIER_INTERNAL_BASE + 2,
   RGP_BARRIER_INTERNAL_PRE_COPY_QUERY_POOL_RESULTS_SYNC = RGP_BARRIER_INTERNAL_BASE + 3
};

void radv_describe_begin_cmd_buffer(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_end_cmd_buffer(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_draw(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_dispatch(struct radv_cmd_buffer *cmd_buffer, const struct radv_dispatch_info *info);
void radv_describe_begin_render_pass_clear(struct radv_cmd_buffer *cmd_buffer, VkImageAspectFlagBits aspects);
void radv_describe_end_render_pass_clear(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_begin_render_pass_resolve(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_end_render_pass_resolve(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_barrier_start(struct radv_cmd_buffer *cmd_buffer, enum rgp_barrier_reason reason);
void radv_describe_barrier_end(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_barrier_end_delayed(struct radv_cmd_buffer *cmd_buffer);
void radv_describe_layout_transition(struct radv_cmd_buffer *cmd_buffer, const struct radv_barrier_data *barrier);
void radv_describe_begin_accel_struct_build(struct radv_cmd_buffer *cmd_buffer, uint32_t count);
void radv_describe_end_accel_struct_build(struct radv_cmd_buffer *cmd_buffer);

void radv_sqtt_emit_relocated_shaders(struct radv_cmd_buffer *cmd_buffer, struct radv_graphics_pipeline *pipeline);

void radv_write_user_event_marker(struct radv_cmd_buffer *cmd_buffer, enum rgp_sqtt_marker_user_event_type type,
                                  const char *str);

static inline uint32_t
radv_conv_prim_to_gs_out(uint32_t topology, bool is_ngg)
{
   switch (topology) {
   case V_008958_DI_PT_POINTLIST:
   case V_008958_DI_PT_PATCH:
      return V_028A6C_POINTLIST;
   case V_008958_DI_PT_LINELIST:
   case V_008958_DI_PT_LINESTRIP:
   case V_008958_DI_PT_LINELIST_ADJ:
   case V_008958_DI_PT_LINESTRIP_ADJ:
      return V_028A6C_LINESTRIP;
   case V_008958_DI_PT_TRILIST:
   case V_008958_DI_PT_TRISTRIP:
   case V_008958_DI_PT_TRIFAN:
   case V_008958_DI_PT_TRILIST_ADJ:
   case V_008958_DI_PT_TRISTRIP_ADJ:
      return V_028A6C_TRISTRIP;
   case V_008958_DI_PT_RECTLIST:
      return is_ngg ? V_028A6C_RECTLIST : V_028A6C_TRISTRIP;
   default:
      assert(0);
      return 0;
   }
}

static inline uint32_t
radv_translate_prim(unsigned topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return V_008958_DI_PT_POINTLIST;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return V_008958_DI_PT_LINELIST;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return V_008958_DI_PT_LINESTRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return V_008958_DI_PT_TRILIST;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return V_008958_DI_PT_TRISTRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return V_008958_DI_PT_TRIFAN;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      return V_008958_DI_PT_LINELIST_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
      return V_008958_DI_PT_LINESTRIP_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      return V_008958_DI_PT_TRILIST_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
      return V_008958_DI_PT_TRISTRIP_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
      return V_008958_DI_PT_PATCH;
   default:
      unreachable("unhandled primitive type");
   }
}

static inline bool
radv_prim_is_points_or_lines(unsigned topology)
{
   switch (topology) {
   case V_008958_DI_PT_POINTLIST:
   case V_008958_DI_PT_LINELIST:
   case V_008958_DI_PT_LINESTRIP:
   case V_008958_DI_PT_LINELIST_ADJ:
   case V_008958_DI_PT_LINESTRIP_ADJ:
      return true;
   default:
      return false;
   }
}

static inline bool
radv_rast_prim_is_point(unsigned rast_prim)
{
   return rast_prim == V_028A6C_POINTLIST;
}

static inline bool
radv_rast_prim_is_line(unsigned rast_prim)
{
   return rast_prim == V_028A6C_LINESTRIP;
}

static inline bool
radv_rast_prim_is_points_or_lines(unsigned rast_prim)
{
   return radv_rast_prim_is_point(rast_prim) || radv_rast_prim_is_line(rast_prim);
}

static inline bool
radv_polygon_mode_is_point(unsigned polygon_mode)
{
   return polygon_mode == V_028814_X_DRAW_POINTS;
}

static inline bool
radv_polygon_mode_is_line(unsigned polygon_mode)
{
   return polygon_mode == V_028814_X_DRAW_LINES;
}

static inline bool
radv_polygon_mode_is_points_or_lines(unsigned polygon_mode)
{
   return radv_polygon_mode_is_point(polygon_mode) || radv_polygon_mode_is_line(polygon_mode);
}

static inline bool
radv_primitive_topology_is_line_list(unsigned primitive_topology)
{
   return primitive_topology == V_008958_DI_PT_LINELIST || primitive_topology == V_008958_DI_PT_LINELIST_ADJ;
}

static inline unsigned
radv_get_num_vertices_per_prim(const struct radv_graphics_state_key *gfx_state)
{
   if (gfx_state->ia.topology == V_008958_DI_PT_NONE) {
      /* When the topology is unknown (with graphics pipeline library), return the maximum number of
       * vertices per primitives for VS. This is used to lower NGG (the HW will ignore the extra
       * bits for points/lines) and also to enable NGG culling unconditionally (it will be disabled
       * dynamically for points/lines).
       */
      return 3;
   } else {
      /* Need to add 1, because: V_028A6C_POINTLIST=0, V_028A6C_LINESTRIP=1, V_028A6C_TRISTRIP=2, etc. */
      return radv_conv_prim_to_gs_out(gfx_state->ia.topology, false) + 1;
   }
}

uint32_t radv_get_vgt_gs_out(struct radv_shader **shaders, uint32_t primitive_topology);

struct radv_vgt_shader_key radv_get_vgt_shader_key(const struct radv_device *device, struct radv_shader **shaders,
                                                   const struct radv_shader *gs_copy_shader);

static inline uint32_t
radv_translate_fill(VkPolygonMode func)
{
   switch (func) {
   case VK_POLYGON_MODE_FILL:
      return V_028814_X_DRAW_TRIANGLES;
   case VK_POLYGON_MODE_LINE:
      return V_028814_X_DRAW_LINES;
   case VK_POLYGON_MODE_POINT:
      return V_028814_X_DRAW_POINTS;
   default:
      assert(0);
      return V_028814_X_DRAW_POINTS;
   }
}

static inline uint32_t
radv_translate_stencil_op(enum VkStencilOp op)
{
   switch (op) {
   case VK_STENCIL_OP_KEEP:
      return V_02842C_STENCIL_KEEP;
   case VK_STENCIL_OP_ZERO:
      return V_02842C_STENCIL_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return V_02842C_STENCIL_REPLACE_TEST;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return V_02842C_STENCIL_ADD_CLAMP;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return V_02842C_STENCIL_SUB_CLAMP;
   case VK_STENCIL_OP_INVERT:
      return V_02842C_STENCIL_INVERT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return V_02842C_STENCIL_ADD_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return V_02842C_STENCIL_SUB_WRAP;
   default:
      return 0;
   }
}

static inline uint32_t
radv_translate_blend_logic_op(VkLogicOp op)
{
   switch (op) {
   case VK_LOGIC_OP_CLEAR:
      return V_028808_ROP3_CLEAR;
   case VK_LOGIC_OP_AND:
      return V_028808_ROP3_AND;
   case VK_LOGIC_OP_AND_REVERSE:
      return V_028808_ROP3_AND_REVERSE;
   case VK_LOGIC_OP_COPY:
      return V_028808_ROP3_COPY;
   case VK_LOGIC_OP_AND_INVERTED:
      return V_028808_ROP3_AND_INVERTED;
   case VK_LOGIC_OP_NO_OP:
      return V_028808_ROP3_NO_OP;
   case VK_LOGIC_OP_XOR:
      return V_028808_ROP3_XOR;
   case VK_LOGIC_OP_OR:
      return V_028808_ROP3_OR;
   case VK_LOGIC_OP_NOR:
      return V_028808_ROP3_NOR;
   case VK_LOGIC_OP_EQUIVALENT:
      return V_028808_ROP3_EQUIVALENT;
   case VK_LOGIC_OP_INVERT:
      return V_028808_ROP3_INVERT;
   case VK_LOGIC_OP_OR_REVERSE:
      return V_028808_ROP3_OR_REVERSE;
   case VK_LOGIC_OP_COPY_INVERTED:
      return V_028808_ROP3_COPY_INVERTED;
   case VK_LOGIC_OP_OR_INVERTED:
      return V_028808_ROP3_OR_INVERTED;
   case VK_LOGIC_OP_NAND:
      return V_028808_ROP3_NAND;
   case VK_LOGIC_OP_SET:
      return V_028808_ROP3_SET;
   default:
      unreachable("Unhandled logic op");
   }
}

static inline uint32_t
radv_translate_blend_function(VkBlendOp op)
{
   switch (op) {
   case VK_BLEND_OP_ADD:
      return V_028780_COMB_DST_PLUS_SRC;
   case VK_BLEND_OP_SUBTRACT:
      return V_028780_COMB_SRC_MINUS_DST;
   case VK_BLEND_OP_REVERSE_SUBTRACT:
      return V_028780_COMB_DST_MINUS_SRC;
   case VK_BLEND_OP_MIN:
      return V_028780_COMB_MIN_DST_SRC;
   case VK_BLEND_OP_MAX:
      return V_028780_COMB_MAX_DST_SRC;
   default:
      return 0;
   }
}

static inline uint32_t
radv_translate_blend_factor(enum amd_gfx_level gfx_level, VkBlendFactor factor)
{
   switch (factor) {
   case VK_BLEND_FACTOR_ZERO:
      return V_028780_BLEND_ZERO;
   case VK_BLEND_FACTOR_ONE:
      return V_028780_BLEND_ONE;
   case VK_BLEND_FACTOR_SRC_COLOR:
      return V_028780_BLEND_SRC_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
      return V_028780_BLEND_ONE_MINUS_SRC_COLOR;
   case VK_BLEND_FACTOR_DST_COLOR:
      return V_028780_BLEND_DST_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR:
      return V_028780_BLEND_ONE_MINUS_DST_COLOR;
   case VK_BLEND_FACTOR_SRC_ALPHA:
      return V_028780_BLEND_SRC_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
      return V_028780_BLEND_ONE_MINUS_SRC_ALPHA;
   case VK_BLEND_FACTOR_DST_ALPHA:
      return V_028780_BLEND_DST_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
      return V_028780_BLEND_ONE_MINUS_DST_ALPHA;
   case VK_BLEND_FACTOR_CONSTANT_COLOR:
      return gfx_level >= GFX11 ? V_028780_BLEND_CONSTANT_COLOR_GFX11 : V_028780_BLEND_CONSTANT_COLOR_GFX6;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
      return gfx_level >= GFX11 ? V_028780_BLEND_ONE_MINUS_CONSTANT_COLOR_GFX11
                                : V_028780_BLEND_ONE_MINUS_CONSTANT_COLOR_GFX6;
   case VK_BLEND_FACTOR_CONSTANT_ALPHA:
      return gfx_level >= GFX11 ? V_028780_BLEND_CONSTANT_ALPHA_GFX11 : V_028780_BLEND_CONSTANT_ALPHA_GFX6;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
      return gfx_level >= GFX11 ? V_028780_BLEND_ONE_MINUS_CONSTANT_ALPHA_GFX11
                                : V_028780_BLEND_ONE_MINUS_CONSTANT_ALPHA_GFX6;
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:
      return V_028780_BLEND_SRC_ALPHA_SATURATE;
   case VK_BLEND_FACTOR_SRC1_COLOR:
      return gfx_level >= GFX11 ? V_028780_BLEND_SRC1_COLOR_GFX11 : V_028780_BLEND_SRC1_COLOR_GFX6;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
      return gfx_level >= GFX11 ? V_028780_BLEND_INV_SRC1_COLOR_GFX11 : V_028780_BLEND_INV_SRC1_COLOR_GFX6;
   case VK_BLEND_FACTOR_SRC1_ALPHA:
      return gfx_level >= GFX11 ? V_028780_BLEND_SRC1_ALPHA_GFX11 : V_028780_BLEND_SRC1_ALPHA_GFX6;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
      return gfx_level >= GFX11 ? V_028780_BLEND_INV_SRC1_ALPHA_GFX11 : V_028780_BLEND_INV_SRC1_ALPHA_GFX6;
   default:
      return 0;
   }
}

static inline uint32_t
radv_translate_blend_opt_factor(VkBlendFactor factor, bool is_alpha)
{
   switch (factor) {
   case VK_BLEND_FACTOR_ZERO:
      return V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_ALL;
   case VK_BLEND_FACTOR_ONE:
      return V_028760_BLEND_OPT_PRESERVE_ALL_IGNORE_NONE;
   case VK_BLEND_FACTOR_SRC_COLOR:
      return is_alpha ? V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0 : V_028760_BLEND_OPT_PRESERVE_C1_IGNORE_C0;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
      return is_alpha ? V_028760_BLEND_OPT_PRESERVE_A0_IGNORE_A1 : V_028760_BLEND_OPT_PRESERVE_C0_IGNORE_C1;
   case VK_BLEND_FACTOR_SRC_ALPHA:
      return V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
      return V_028760_BLEND_OPT_PRESERVE_A0_IGNORE_A1;
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:
      return is_alpha ? V_028760_BLEND_OPT_PRESERVE_ALL_IGNORE_NONE : V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_A0;
   default:
      return V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_NONE;
   }
}

static inline uint32_t
radv_translate_blend_opt_function(VkBlendOp op)
{
   switch (op) {
   case VK_BLEND_OP_ADD:
      return V_028760_OPT_COMB_ADD;
   case VK_BLEND_OP_SUBTRACT:
      return V_028760_OPT_COMB_SUBTRACT;
   case VK_BLEND_OP_REVERSE_SUBTRACT:
      return V_028760_OPT_COMB_REVSUBTRACT;
   case VK_BLEND_OP_MIN:
      return V_028760_OPT_COMB_MIN;
   case VK_BLEND_OP_MAX:
      return V_028760_OPT_COMB_MAX;
   default:
      return V_028760_OPT_COMB_BLEND_DISABLED;
   }
}

static inline bool
radv_blend_factor_uses_dst(VkBlendFactor factor)
{
   return factor == VK_BLEND_FACTOR_DST_COLOR || factor == VK_BLEND_FACTOR_DST_ALPHA ||
          factor == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE || factor == VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA ||
          factor == VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
}

static inline bool
radv_is_dual_src(VkBlendFactor factor)
{
   switch (factor) {
   case VK_BLEND_FACTOR_SRC1_COLOR:
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
   case VK_BLEND_FACTOR_SRC1_ALPHA:
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
      return true;
   default:
      return false;
   }
}

static ALWAYS_INLINE bool
radv_can_enable_dual_src(const struct vk_color_blend_attachment_state *att)
{
   VkBlendOp eqRGB = att->color_blend_op;
   VkBlendFactor srcRGB = att->src_color_blend_factor;
   VkBlendFactor dstRGB = att->dst_color_blend_factor;
   VkBlendOp eqA = att->alpha_blend_op;
   VkBlendFactor srcA = att->src_alpha_blend_factor;
   VkBlendFactor dstA = att->dst_alpha_blend_factor;
   bool eqRGB_minmax = eqRGB == VK_BLEND_OP_MIN || eqRGB == VK_BLEND_OP_MAX;
   bool eqA_minmax = eqA == VK_BLEND_OP_MIN || eqA == VK_BLEND_OP_MAX;

   if (!eqRGB_minmax && (radv_is_dual_src(srcRGB) || radv_is_dual_src(dstRGB)))
      return true;
   if (!eqA_minmax && (radv_is_dual_src(srcA) || radv_is_dual_src(dstA)))
      return true;
   return false;
}

static inline void
radv_normalize_blend_factor(VkBlendOp op, VkBlendFactor *src_factor, VkBlendFactor *dst_factor)
{
   if (op == VK_BLEND_OP_MIN || op == VK_BLEND_OP_MAX) {
      *src_factor = VK_BLEND_FACTOR_ONE;
      *dst_factor = VK_BLEND_FACTOR_ONE;
   }
}

void radv_blend_remove_dst(VkBlendOp *func, VkBlendFactor *src_factor, VkBlendFactor *dst_factor,
                           VkBlendFactor expected_dst, VkBlendFactor replacement_src);

ALWAYS_INLINE static bool
radv_is_streamout_enabled(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_streamout_state *so = &cmd_buffer->state.streamout;

   /* Streamout must be enabled for the PRIMITIVES_GENERATED query to work. */
   return (so->streamout_enabled || cmd_buffer->state.active_prims_gen_queries) && !cmd_buffer->state.suspend_streamout;
}

/*
 * Queue helper to get ring.
 * placed here as it needs queue + device structs.
 */
static inline enum amd_ip_type
radv_queue_ring(const struct radv_queue *queue)
{
   struct radv_device *device = radv_queue_device(queue);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   return radv_queue_family_to_ring(pdev, queue->state.qf);
}

/**
 * Helper used for debugging compiler issues by enabling/disabling LLVM for a
 * specific shader stage (developers only).
 */
static inline bool
radv_use_llvm_for_stage(const struct radv_physical_device *pdev, UNUSED gl_shader_stage stage)
{
   return pdev->use_llvm;
}

static inline bool
radv_has_shader_buffer_float_minmax(const struct radv_physical_device *pdev, unsigned bitsize)
{
   return (pdev->info.gfx_level <= GFX7 && !pdev->use_llvm) || pdev->info.gfx_level == GFX10 ||
          pdev->info.gfx_level == GFX10_3 || (pdev->info.gfx_level == GFX11 && bitsize == 32);
}

static inline bool
radv_has_pops(const struct radv_physical_device *pdev)
{
   return pdev->info.gfx_level >= GFX9 && !pdev->use_llvm;
}

unsigned radv_compact_spi_shader_col_format(const struct radv_shader *ps, uint32_t spi_shader_col_format);

/* radv_spm.c */
bool radv_spm_init(struct radv_device *device);
void radv_spm_finish(struct radv_device *device);
void radv_emit_spm_setup(struct radv_device *device, struct radeon_cmdbuf *cs, enum radv_queue_family qf);

void radv_destroy_graphics_pipeline(struct radv_device *device, struct radv_graphics_pipeline *pipeline);
void radv_destroy_graphics_lib_pipeline(struct radv_device *device, struct radv_graphics_lib_pipeline *pipeline);
void radv_destroy_compute_pipeline(struct radv_device *device, struct radv_compute_pipeline *pipeline);
void radv_destroy_ray_tracing_pipeline(struct radv_device *device, struct radv_ray_tracing_pipeline *pipeline);

void radv_begin_conditional_rendering(struct radv_cmd_buffer *cmd_buffer, uint64_t va, bool draw_visible);
void radv_end_conditional_rendering(struct radv_cmd_buffer *cmd_buffer);

bool radv_gang_init(struct radv_cmd_buffer *cmd_buffer);
void radv_gang_cache_flush(struct radv_cmd_buffer *cmd_buffer);

static inline bool
radv_uses_device_generated_commands(const struct radv_device *device)
{
   return device->vk.enabled_features.deviceGeneratedCommands || device->vk.enabled_features.deviceGeneratedCompute;
}

static inline bool
radv_uses_primitives_generated_query(const struct radv_device *device)
{
   return device->vk.enabled_features.primitivesGeneratedQuery ||
          device->vk.enabled_features.primitivesGeneratedQueryWithRasterizerDiscard ||
          device->vk.enabled_features.primitivesGeneratedQueryWithNonZeroStreams;
}

static inline bool
radv_uses_image_float32_atomics(const struct radv_device *device)
{
   return device->vk.enabled_features.shaderImageFloat32Atomics ||
          device->vk.enabled_features.sparseImageFloat32Atomics ||
          device->vk.enabled_features.shaderImageFloat32AtomicMinMax ||
          device->vk.enabled_features.sparseImageFloat32AtomicMinMax;
}

struct radv_compute_pipeline_metadata {
   uint32_t shader_va;
   uint32_t rsrc1;
   uint32_t rsrc2;
   uint32_t rsrc3;
   uint32_t compute_resource_limits;
   uint32_t block_size_x;
   uint32_t block_size_y;
   uint32_t block_size_z;
   uint32_t wave32;
   uint32_t grid_base_sgpr;
   uint32_t push_const_sgpr;
   uint64_t inline_push_const_mask;
};

void radv_get_compute_pipeline_metadata(const struct radv_device *device, const struct radv_compute_pipeline *pipeline,
                                        struct radv_compute_pipeline_metadata *metadata);

#define RADV_FROM_HANDLE(__radv_type, __name, __handle) VK_FROM_HANDLE(__radv_type, __name, __handle)

VK_DEFINE_HANDLE_CASTS(radv_cmd_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER)
VK_DEFINE_HANDLE_CASTS(radv_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_DEFINE_HANDLE_CASTS(radv_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(radv_physical_device, vk.base, VkPhysicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(radv_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)
VK_DEFINE_NONDISP_HANDLE_CASTS(radv_device_memory, base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(radv_pipeline, base, VkPipeline, VK_OBJECT_TYPE_PIPELINE)
VK_DEFINE_NONDISP_HANDLE_CASTS(radv_shader_object, base, VkShaderEXT, VK_OBJECT_TYPE_SHADER_EXT);

static inline uint64_t
radv_get_tdr_timeout_for_ip(enum amd_ip_type ip_type)
{
   const uint64_t compute_tdr_duration_ns = 60000000000ull; /* 1 minute (default in kernel) */
   const uint64_t other_tdr_duration_ns = 10000000000ull;   /* 10 seconds (default in kernel) */

   return ip_type == AMD_IP_COMPUTE ? compute_tdr_duration_ns : other_tdr_duration_ns;
}

#ifdef __cplusplus
}
#endif

#endif /* RADV_PRIVATE_H */
