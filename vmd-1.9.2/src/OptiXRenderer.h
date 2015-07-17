/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
* RCS INFORMATION:
*
*      $RCSfile: OptiXDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.57 $         $Date: 2014/12/21 19:36:00 $
*
***************************************************************************
* DESCRIPTION:
*   VMD built-in Tachyon/OptiX renderer implementation.
*
* This work is described in:
*  "GPU-Accelerated Molecular Visualization on
*   Petascale Supercomputing Platforms"
*   John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
*   UltraVis'13: Proceedings of the 8th International Workshop on
*   Ultrascale Visualization, pp. 6:1-6:8, 2013.
*   http://dx.doi.org/10.1145/2535571.2535595
*
* Significant portions of this code are derived from Tachyon:
*   "An Efficient Library for Parallel Ray Tracing and Animation"
*   John E. Stone.  Master's Thesis, University of Missouri-Rolla,
*   Department of Computer Science, April 1998
*
*   "Rendering of Numerical Flow Simulations Using MPI"
*   John Stone and Mark Underwood.
*   Second MPI Developers Conference, pages 138-141, 1996.
*   http://dx.doi.org/10.1109/MPIDC.1996.534105
*
***************************************************************************/

#ifndef LIBOPTIXRENDERER
#define LIBOPTIXRENDERER

#include <stdio.h>
#include <stdlib.h>
#include <optix.h>
#include <optix_math.h>
#include "Matrix4.h"
#include "ResizeArray.h"
#include "WKFUtils.h"

#if defined(VMDOPTIX_INTERACTIVE_OPENGL)
#include "glwin.h"
#endif

/// structure containing material properties used to shade a Displayable
typedef struct {
  RTmaterial mat; 
  int isvalid;
  float ambient;
  float diffuse;
  float specular;
  float shininess;
  float reflectivity;
  float opacity;
  float outline;
  float outlinewidth;
  int transmode;
  int ind;
} ort_material;

typedef struct {
  float dir[3];
  float color[3]; // XXX ignored for now
} ort_directional_light;

class OptiXRenderer {
public: 
  enum FogMode { RT_FOG_NONE=0, RT_FOG_LINEAR=1, RT_FOG_EXP=2, RT_FOG_EXP2=3 };
  enum CameraProjection { RT_PERSPECTIVE=0, RT_ORTHOGRAPHIC=1 };
  enum Verbosity { RT_VERB_MIN=0, RT_VERB_TIMING=1, RT_VERB_DEBUG=2 };
  enum BGMode { RT_BACKGROUND_TEXTURE_SOLID=0,
                RT_BACKGROUND_TEXTURE_SKY_SPHERE=1,
                RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE=2 };
  enum RayType { RT_RAY_TYPE_RADIANCE=0,   ///< normal radiance rays
                 RT_RAY_TYPE_SHADOW=1,     ///< shadow probe/AO rays
                 RT_RAY_TYPE_COUNT=2 };    ///< total count of ray types
  enum RayGen  { RT_RAY_GEN_CLEAR_RNG_BUFFERS=0,
                 RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER=1,
                 RT_RAY_GEN_CLEAR_FRAMEBUFFER=2,
                 RT_RAY_GEN_ACCUMULATE=3,  ///< a render pass to accum buf
                 RT_RAY_GEN_COPY_FINISH=4, ///< copy accum buf to framebuffer
                 RT_RAY_GEN_COUNT=5 };     ///< total count of ray gen pgms

private:
  Verbosity verbose;                      ///< console perf/debugging output
  wkf_timerhandle ort_timer;              ///< general purpose timer
  int width;                              ///< image width in pixels
  int height;                             ///< image height in pixels
  char shaderpath[8192];                  ///< path to OptiX shader PTX file

  // OptiX objects managed by VMD
  int context_created;                    ///< flag when context is valid
  RTcontext context;                      ///< OptiX main context
  RTresult lasterror;                     ///< Last OptiX error code if any
  RTacceleration acceleration;            ///< OptiX AS for scene

  int buffers_allocated;                  ///< flag for buffer state
  RTbuffer framebuffer;                   ///< output image buffer
  RTvariable framebuffer_v;               ///< output image buffer variable
  RTbuffer accumulation_buffer;           ///< intermediate GPU-local accum buf
  RTvariable accumulation_buffer_v;       ///< accum buffer variable
  RTvariable progressive_index_v;         ///< VCA progressive subframe index
#ifndef VMDOPTIX_VCA
  RTbuffer aa_rng_buffer;                 ///< intermediate GPU-local AA RNG buf
  RTvariable aa_rng_buffer_v;             ///< AA RNG buffer variable
  RTbuffer ao_rng_buffer;                 ///< intermediate GPU-local AO RNG buf
  RTvariable ao_rng_buffer_v;             ///< AO RNG buffer variable
#endif

#if defined(VMDOPTIX_VCA)
  RTvariable light_list_v;                ///< list of lights
#endif
  RTvariable lightbuffer_v;               ///< list of lights
  RTbuffer lightbuffer;                   ///< list of lights

  RTvariable ao_ambient_v;                ///< AO ambient lighting scalefactor
  float ao_ambient;                       ///< AO ambient lighting scalefactor
  RTvariable ao_direct_v;                 ///< AO direct lighting scalefactor
  float ao_direct;                        ///< AO direct lighting scalefactor

  RTprogram clear_rng_buffers_pgm;         ///< clear AA and AO RNG buffers
  RTprogram clear_accumulation_buffer_pgm; ///< clear accum buffer
  RTprogram clear_framebuffer_pgm;         ///< clear framebuffer
  RTprogram draw_accumulation_buffer_pgm;  ///< copy accum buffer to framebuffer

  int ray_gen_pgms_registered;                 ///< flag ray gen pgm reg state
  RTprogram ray_gen_pgm_perspective;           ///< perspective cam (non-stereo)
  RTprogram ray_gen_pgm_perspective_dof;       ///< perspective cam (non-stereo)
  RTprogram ray_gen_pgm_perspective_stereo;    ///< perspective cam (stereo)
  RTprogram ray_gen_pgm_perspective_stereo_dof; //< perspective cam (stereo)

  RTprogram ray_gen_pgm_orthographic;        ///< orthographic cam (non-stereo)
  RTprogram ray_gen_pgm_orthographic_stereo; ///< orthographic cam (stereo)

  RTprogram closest_hit_pgm_general;       ///< fully general shader
  RTprogram closest_hit_pgm_special[64];   ///< 2^6 template-specialized fctns 

  RTprogram any_hit_pgm_opaque;            ///< shadows for opaque objects
  RTprogram any_hit_pgm_transmission;      ///< shadows with filtering

  RTprogram miss_pgm;                      ///< miss/background shader

  RTmaterial material_general;             ///< fully-general material
  RTmaterial material_special[64];         ///< 2^6 specialized materials
  int material_special_counts[64];         ///< usage count in current scene


  // cylinder array primitive
  RTprogram cylinder_array_isct_pgm;       ///< cylinder array intersection code
  RTprogram cylinder_array_bbox_pgm;       ///< cylinder array bounding box code
  long cylinder_array_cnt;                 ///< number of cylinder in scene

  // color-per-cylinder array primitive
  RTprogram cylinder_array_color_isct_pgm; ///< cylinder array intersection code
  RTprogram cylinder_array_color_bbox_pgm; ///< cylinder array bounding box code
  long cylinder_array_color_cnt;           ///< number of cylinders in scene


  // color-per-ring array primitive
  RTprogram ring_array_color_isct_pgm;    ///< ring array intersection code
  RTprogram ring_array_color_bbox_pgm;    ///< ring array bounding box code
  long ring_array_color_cnt;              ///< number of rings in scene


  // sphere array primitive
  RTprogram sphere_array_isct_pgm;        ///< sphere array intersection code
  RTprogram sphere_array_bbox_pgm;        ///< sphere array bounding box code
  long sphere_array_cnt;                  ///< number of spheres in scene

  // color-per-sphere array primitive
  RTprogram sphere_array_color_isct_pgm;  ///< sphere array intersection code
  RTprogram sphere_array_color_bbox_pgm;  ///< sphere array bounding box code
  long sphere_array_color_cnt;            ///< number of spheres in scene


  // triangle mesh primitives of various types
  RTprogram tricolor_isct_pgm;            ///< triangle mesh intersection code
  RTprogram tricolor_bbox_pgm;            ///< triangle mesh bounding box code
  long tricolor_cnt;                      ///< number of triangles scene

  RTprogram trimesh_c4u_n3b_v3f_isct_pgm; ///< triangle mesh intersection code
  RTprogram trimesh_c4u_n3b_v3f_bbox_pgm; ///< triangle mesh bounding box code
  long trimesh_c4u_n3b_v3f_cnt;           ///< number of triangles scene

  RTprogram trimesh_n3f_v3f_isct_pgm;     ///< triangle mesh intersection code
  RTprogram trimesh_n3f_v3f_bbox_pgm;     ///< triangle mesh bounding box code
  long trimesh_n3b_v3f_cnt;               ///< number of triangles scene

  RTprogram trimesh_n3b_v3f_isct_pgm;     ///< triangle mesh intersection code
  RTprogram trimesh_n3b_v3f_bbox_pgm;     ///< triangle mesh bounding box code
  long trimesh_n3f_v3f_cnt;               ///< number of triangles scene

  RTgroup         root_group;
  RTvariable      root_object;
  RTvariable      root_shadower;
  RTacceleration  root_acceleration;

  //
  // OptiX shader state variables and the like
  //

  // shadow rendering mode 
  RTvariable shadows_enabled_v;         ///< shadow enable/disable flag  
  int shadows_enabled;                  ///< shadow enable/disable flag  

  RTvariable cam_zoom_v;                ///< camera zoom factor
  float cam_zoom;                       ///< camera zoom factor
  
  RTvariable cam_pos_v;                 ///< camera location
  RTvariable cam_U_v;                   ///< camera basis vector
  RTvariable cam_V_v;                   ///< camera basis vector
  RTvariable cam_W_v;                   ///< camera basis vector

  RTvariable cam_stereo_eyesep_v;           ///< stereo eye separation
  float cam_stereo_eyesep;                  ///< stereo eye separation
  RTvariable cam_stereo_convergence_dist_v; ///< stereo convergence distance
  float cam_stereo_convergence_dist;        ///< stereo convergence distance

  int dof_enabled;                      ///< DoF enable/disable flag  
  RTvariable cam_dof_focal_dist_v;      ///< DoF focal distance
  RTvariable cam_dof_aperture_rad_v;    ///< DoF aperture radius for CoC
  float cam_dof_focal_dist;             ///< DoF focal distance
  float cam_dof_fnumber;                ///< DoF f/stop number

  RTvariable camera_projection_v;       ///< camera projection mode
  CameraProjection camera_projection;   ///< camera projection mode

  int ext_aa_loops;                     ///< Multi-pass AA iterations

  RTvariable accum_norm_v;              ///< Accum. buf normalization factor

  RTvariable aa_samples_v;              ///< AA samples per pixel
  int aa_samples;                       ///< AA samples per pixel

  RTvariable ao_samples_v;              ///< AO samples per pixel
  int ao_samples;                       ///< AO samples per pixel

  // background color and/or gradient parameters
  BGMode scene_background_mode;         ///< which miss program to use...
  RTvariable scene_bg_color_v;          ///< background color
  float scene_bg_color[3];              ///< background color
  RTvariable scene_bg_grad_top_v;       ///< background gradient top color
  float scene_bg_grad_top[3];           ///< background gradient top color
  RTvariable scene_bg_grad_bot_v;       ///< background gradient bottom color
  float scene_bg_grad_bot[3];           ///< background gradient bottom color
  RTvariable scene_gradient_v;          ///< background gradient vector
  float scene_gradient[3];              ///< background gradient vector
  RTvariable scene_gradient_topval_v;   ///< background gradient top value
  float scene_gradient_topval;          ///< background gradient top value
  RTvariable scene_gradient_botval_v;   ///< background gradient bot value
  float scene_gradient_botval;          ///< background gradient bot value
  RTvariable scene_gradient_invrange_v; ///< background gradient rcp range
  float scene_gradient_invrange;        ///< background gradient rcp range

  // fog / depth cueing parameters
  RTvariable fog_mode_v;                ///< fog mode
  int fog_mode;                         ///< fog mode
  RTvariable fog_start_v;               ///< fog start
  float fog_start;                      ///< fog start
  RTvariable fog_end_v;                 ///< fog end
  float fog_end;                        ///< fog end
  RTvariable fog_density_v;             ///< fog density
  float fog_density;                    ///< fog density

  // cache VMD material values
  ResizeArray<ort_material> materialcache;

  // list of directional lights
  ResizeArray<ort_directional_light> directional_lights;

  // keep track of all of the OptiX objects we create on-the-fly...
  ResizeArray<RTgeometry> geomlist;
  ResizeArray<RTgeometryinstance> geominstancelist;
  ResizeArray<RTbuffer> bufferlist;

public:
  OptiXRenderer(void);
  ~OptiXRenderer(void);

  /// static methods for querying OptiX-supported GPU hardware independent
  /// of whether we actually have an active context.
  static unsigned int device_list(int **, char ***);
  static unsigned int device_count(void);
  static unsigned int optix_version(void);

  /// initialize the OptiX context 
  void setup_context(int width, int height);

  /// shadows
  void shadows_on(int onoff) { shadows_enabled = (onoff != 0); }

  /// antialiasing (samples > 1 == on)
  void set_aa_samples(int cnt) { aa_samples = cnt; }

  /// set the camera projection mode
  void set_camera_projection(CameraProjection m) { camera_projection = m; }

  /// set camera zoom factor
  void set_camera_zoom(float zoomfactor) { cam_zoom = zoomfactor; }

  /// set stereo eye separation
  void set_camera_stereo_eyesep(float eyesep) { cam_stereo_eyesep = eyesep; }
  
  /// set stereo convergence distance
  void set_camera_stereo_convergence_dist(float dist) {
    cam_stereo_convergence_dist = dist;
  }

  /// depth of field on/off
  void dof_on(int onoff) { dof_enabled = (onoff != 0); }

  /// set depth of field focal plane distance
  void set_camera_dof_focal_dist(float d) { cam_dof_focal_dist = d; }

  /// set depth of field f/stop number
  void set_camera_dof_fnumber(float n) { cam_dof_fnumber = n; }

  /// ambient occlusion (samples > 1 == on)
  void set_ao_samples(int cnt) { ao_samples = cnt; }

  /// set AO ambient lighting factor
  void set_ao_ambient(float aoa) { ao_ambient = aoa; }

  /// set AO direct lighting factor
  void set_ao_direct(float aod) { ao_direct = aod; }

  void set_bg_mode(BGMode m) { scene_background_mode = m; }
  void set_bg_color(float *rgb) { memcpy(scene_bg_color, rgb, sizeof(scene_bg_color)); }
  void set_bg_color_grad_top(float *rgb) { memcpy(scene_bg_grad_top, rgb, sizeof(scene_bg_grad_top)); }
  void set_bg_color_grad_bot(float *rgb) { memcpy(scene_bg_grad_bot, rgb, sizeof(scene_bg_grad_bot)); }
  void set_bg_gradient(float *vec) { memcpy(scene_gradient, vec, sizeof(scene_gradient)); }
  void set_bg_gradient_topval(float v) { scene_gradient_topval = v; }
  void set_bg_gradient_botval(float v) { scene_gradient_botval = v; }

  void set_cue_mode(FogMode mode, float start, float end, float density) {
    fog_mode = mode;
    fog_start = start;
    fog_end = end;
    fog_density = density;
  }

  void init_materials();
  void add_material(int matindex, float ambient, float diffuse,
                    float specular, float shininess, float reflectivity,
                    float opacity, float outline, float outlinewidth, 
                    int transmode);
  void set_material(RTgeometryinstance instance, int matindex, float *uniform_color);

  void clear_all_lights() { directional_lights.clear(); }
  void add_directional_light(const float *dir, const float *color);

  void update_rendering_state(void);

  void allocate_framebuffer(int fbwidth, int fbheight);
  void resize_framebuffer(int fbwidth, int fbheight);
  void destroy_framebuffer(void);

  void render_compile_and_validate(void);
  void render_to_file(const char *filename); 
#if defined(VMDOPTIX_INTERACTIVE_OPENGL)
  void render_to_glwin(const char *filename);
#endif

  void destroy_context(void);

  void cylinder_array(Matrix4 *wtrans, float rscale, float *uniform_color,
                      int cylnum, float *points, int matindex);

  void cylinder_array_color(Matrix4 *wtrans, float rscale, int cylnum, 
                            float *points, float *radii, float *colors,
                            int matindex);

  void ring_array_color(Matrix4 & wtrans, float rscale, int rnum, 
                        float *centers, float *norms, float *radii, 
                        float *colors, int matindex);

  void sphere_array(Matrix4 *wtrans, float rscale, float *uniform_color,
                    int spnum, float *centers, float *radii, int matindex);

  void sphere_array_color(Matrix4 & wtrans, float rscale, int spnum, 
                          float *centers, float *radii, float *colors, 
                          int matindex);

  void tricolor_list(Matrix4 & wtrans, int numtris, float *vnc, int matindex);

  void trimesh_c4n3v3(Matrix4 & wtrans, int numverts,
                      float *cnv, int numfacets, int * facets, int matindex);

  void trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, char *n, 
                           float *v, int numfacets, int matindex);

  void trimesh_c4u_n3f_v3f(Matrix4 & wtrans, unsigned char *c, 
                           float *n, float *v, int numfacets, int matindex);

  void trimesh_n3b_v3f(Matrix4 & wtrans, float *uniform_color, 
                       char *n, float *v, int numfacets, int matindex);

  void trimesh_n3f_v3f(Matrix4 & wtrans, float *uniform_color, 
                       float *n, float *v, int numfacets, int matindex);

  void tristrip(Matrix4 & wtrans, int numverts, const float * cnv,
                int numstrips, const int *vertsperstrip,
                const int *facets, int matindex);

}; 

#endif

