/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 2013-2014 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
* RCS INFORMATION:
*
*      $RCSfile: OptiXRenderer.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.154 $         $Date: 2014/12/21 19:35:59 $
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

#include <optix.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "Inform.h"
#include "ImageIO.h"
#include "OptiXRenderer.h"
#include "OptiXShaders.h"
#include "Matrix4.h"
#include "utilities.h"
#include "WKFUtils.h"

// enable the interactive ray tracing capability
#if defined(VMDOPTIX_INTERACTIVE_OPENGL)
#include <GL/gl.h>
#endif


// this macro enables or disables the use of an array of
// template-specialized shaders for every combination of
// scene-wide and material-specific shader features.
#define ORT_USE_TEMPLATE_SHADERS 1
#if defined(ORT_USE_TEMPLATE_SHADERS)
static const char *onoffstr(int onoff) {
  return (onoff) ? "on" : "off";
}
#endif


#if 0
// Enable the use of OptiX timeout callbacks to help reduce the likelihood
// of kernel timeouts when rendering on GPUs that are also used for display
#define VMD_ENABLE_OPTIX_TIMEOUTS 1

static int vmd_timeout_init = 0;
static wkf_timerhandle cbtimer;
static float vmd_timeout_lastcallback = 0.0f;

static void vmd_timeout_reset(void) {
  if (vmd_timeout_init == 0) {
    vmd_timeout_init = 1;
    cbtimer = wkf_timer_create();
  }
  wkf_timer_start(cbtimer);
  vmd_timeout_lastcallback = wkf_timer_timenow(cbtimer);
}

static void vmd_timeout_time(float &deltat, float &totalt) {
  double now = wkf_timer_timenow(cbtimer);
  deltat = now - vmd_timeout_lastcallback;
  totalt = now;
  vmd_timeout_lastcallback = now;
}

static int vmd_timeout_cb(void) {
  int earlyexit = 0;
  float deltat, totalt;

  if (vmd_timeout_init == 0) 
    vmd_timeout_reset();

  vmd_timeout_time(deltat, totalt);
  printf("OptiXRenderer) timeout callback: since last %f sec, total %f sec\n",
         deltat, totalt); 
  return earlyexit; 
}

#endif


/* assumes current scope has Context variable named 'context' */
#define RTERR( func )                                                  \
  {                                                                    \
    RTresult code = func;                                              \
    if (code != RT_SUCCESS) {                                          \
      lasterror = code; /* preserve error code for subsequent tests */ \
      const char* message;                                             \
      rtContextGetErrorString(context, code, &message);                \
      msgErr << "OptiXRenderer) ERROR: " << message << " ("            \
             << __FILE__ << ":" << __LINE__ << sendmsg;                \
    }                                                                  \
  }


static unsigned char * cvt_rgb4f_rgb3u(float * rgb4f, int xs, int ys) {
  int rowlen3u = xs*3;
  int sz = xs * ys * 3;
  unsigned char * rgb3u = (unsigned char *) calloc(1, sz);

  int x3u, x4f, y;
  for (y=0; y<ys; y++) {
    int addr3u = y * xs * 3;
    int addr4f = y * xs * 4;
    for (x3u=0,x4f=0; x3u<rowlen3u; x3u+=3,x4f+=4) {
      int tmp;

      tmp = rgb4f[addr4f + x4f    ] * 255.0f;
      rgb3u[addr3u + x3u    ] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = rgb4f[addr4f + x4f + 1] * 255.0f;
      rgb3u[addr3u + x3u + 1] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = rgb4f[addr4f + x4f + 2] * 255.0f;
      rgb3u[addr3u + x3u + 2] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);
    }
  }

  return rgb3u;
}


static unsigned char * cvt_rgb4u_rgb3u(unsigned char * rgb4u, int xs, int ys) {
  int rowlen3u = xs*3;
  int sz = xs * ys * 3;
  unsigned char * rgb3u = (unsigned char *) calloc(1, sz);

  int x3u, x4f, y;
  for (y=0; y<ys; y++) {
    int addr3u = y * xs * 3;
    int addr4f = y * xs * 4;
    for (x3u=0,x4f=0; x3u<rowlen3u; x3u+=3,x4f+=4) {
      rgb3u[addr3u + x3u    ] = rgb4u[addr4f + x4f    ];
      rgb3u[addr3u + x3u + 1] = rgb4u[addr4f + x4f + 1];
      rgb3u[addr3u + x3u + 2] = rgb4u[addr4f + x4f + 2];
    }
  }

  return rgb3u;
}


static int checkfileextension(const char * s, const char * extension) {
  int sz, extsz;
  sz = strlen(s);
  extsz = strlen(extension);

  if (extsz > sz)
    return 0;

  if (!strupncmp(s + (sz - extsz), extension, extsz)) {
    return 1;
  }

  return 0;
}


int OptiXWriteImage(const char* filename, RTbuffer buffer) {
  RTresult result;
  RTsize buffer_width, buffer_height;

  // buffer must be 2-D
  unsigned int bufdim;
  if (rtBufferGetDimensionality(buffer, &bufdim) != RT_SUCCESS) {
    msgErr << "OptiXWriteImage: Failed to get output buffer dimensions!" << sendmsg;
    return -1;
  }

  if (bufdim != 2) {
    msgErr << "OptiXWriteImage: Output buffer is not 2-D!" << sendmsg;
    return -1;
  }

  void * imageData;
  result = rtBufferMap(buffer, &imageData);
  if (result != RT_SUCCESS) {
    RTcontext context;
    const char* error;
    rtBufferGetContext(buffer, &context);
    rtContextGetErrorString(context, result, &error);
    msgErr << "OptiXWriteImage: Error mapping image buffer: " 
           << error << sendmsg;
    return -1;
  }

  // no image data
  if (imageData == NULL) {
    msgErr << "OptiXWriteImage: No image data in output buffer!" << sendmsg;
    return -1;
  }

  result = rtBufferGetSize2D(buffer, &buffer_width, &buffer_height);
  if (result != RT_SUCCESS) {
    // Get error from context
    RTcontext context;
    const char* error;
    rtBufferGetContext(buffer, &context);
    rtContextGetErrorString(context, result, &error);
    msgErr << "OptiX: Error getting dimensions of buffer: " << error << sendmsg;
    return -1;
  }

  RTformat buffer_format;
  if (rtBufferGetFormat(buffer, &buffer_format) != RT_SUCCESS) {
    msgErr << "OptiXWriteImage: failed to query output buffer format!" 
           << sendmsg;
    return -1;
  }

  //
  // convert the format of the final image, and write it to a file
  //
  unsigned char *rgb3u = NULL;
  if (buffer_format == RT_FORMAT_FLOAT4) {
    rgb3u = cvt_rgb4f_rgb3u((float*)imageData, buffer_width, buffer_height);
  } else if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
    rgb3u = cvt_rgb4u_rgb3u((unsigned char*)imageData, buffer_width, buffer_height);
  } else {
    msgErr << "OptiXWriteImage: output buffer format is not rgb4u or rgb4f!" 
           << sendmsg;
    return -1;
  }
  
  result = rtBufferUnmap(buffer);
  if (result != RT_SUCCESS) {
    RTcontext context;
    const char* error;
    rtBufferGetContext(buffer, &context);
    rtContextGetErrorString(context, result, &error);
    msgErr << "OptiXWriteImage: Error unmapping image buffer: " 
           << error << sendmsg;
    return -1;
  }

  if (rgb3u != NULL) {
    int xs = buffer_width;
    int ys = buffer_height;

    FILE *outfile=NULL;
    if ((outfile = fopen(filename, "wb")) == NULL) {
      msgErr << "Could not open file " << filename
             << " in current directory for writing!" << sendmsg;
      free(rgb3u);
      return -1;
    }

    // write the image to a file on disk
    if (checkfileextension(filename, ".bmp")) {
      vmd_writebmp(outfile, rgb3u, xs, ys);
#if defined(VMDPNG)
    } else if (checkfileextension(filename, ".png")) {
      vmd_writepng(outfile, rgb3u, xs, ys);
#endif
    } else if (checkfileextension(filename, ".ppm")) {
      vmd_writeppm(outfile, rgb3u, xs, ys);
    } else if (checkfileextension(filename, ".rgb")) {
      vmd_writergb(outfile, rgb3u, xs, ys);
    } else if (checkfileextension(filename, ".tga")) {
      vmd_writetga(outfile, rgb3u, xs, ys);
    } else {
#if defined(_MSC_VER) || defined(WIN32)
      msgErr << "Unrecognized image file extension, writing Windows Bitmap file."
             << sendmsg;
      vmd_writebmp(outfile, rgb3u, xs, ys);
#else
      msgErr << "Unrecognized image file extension, writing Targa file."
             << sendmsg;
      vmd_writetga(outfile, rgb3u, xs, ys);
#endif
    }
    free(rgb3u);
    fclose(outfile);
  }

  return 0;
}


/// constructor ... initialize some variables
OptiXRenderer::OptiXRenderer() {
  ort_timer = wkf_timer_create(); // create and initialize timer

  // setup path to pre-compiled shader PTX code
  const char *vmddir = getenv("VMDDIR");
  if (vmddir == NULL)
    vmddir = ".";
  sprintf(shaderpath, "%s/shaders/%s", vmddir, "OptiXShaders.ptx");

  // allow runtime override of the default shader path for testing
  if (getenv("VMDOPTIXSHADERPATH"))
    strcpy(shaderpath, getenv("VMDOPTIXSHADERPATH"));

  context_created = 0;         // no context yet
  buffers_allocated = 0;       // flag no buffer allocated yet
  ray_gen_pgms_registered = 0; // ray gen pgms not registered yet

  // set default scene background state
  scene_background_mode = RT_BACKGROUND_TEXTURE_SOLID;
  memset(scene_bg_color, 0, sizeof(scene_bg_color));
  memset(scene_bg_grad_top, 0, sizeof(scene_bg_grad_top));
  memset(scene_bg_grad_bot, 0, sizeof(scene_bg_grad_bot));
  memset(scene_gradient, 0, sizeof(scene_gradient));
  scene_gradient_topval = 1.0f;
  scene_gradient_botval = 0.0f;
  // XXX this has to be recomputed prior to rendering..
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);

  verbose = RT_VERB_MIN;  // keep console quiet except for perf/debugging cases

  // zero out the array of material usage counts for the scene
  memset(material_special_counts, 0, sizeof(material_special_counts));

  cam_zoom = 1.0f;
  cam_stereo_eyesep = 0.06f;
  cam_stereo_convergence_dist = 2.0f;

  shadows_enabled = RT_SHADOWS_OFF; // disable shadows by default 
  aa_samples = 0;            // no AA samples by default

  ao_samples = 0;            // no AO samples by default
  ao_direct = 0.3f;          // AO direct contribution is 30%
  ao_ambient = 0.7f;         // AO ambient contribution is 70%

  dof_enabled = 0;           // disable DoF by default
  cam_dof_focal_dist = 2.0f;
  cam_dof_fnumber = 64.0f;

  fog_mode = RT_FOG_NONE;    // fog/cueing disabled by default
  fog_start = 0.0f;
  fog_end = 10.0f;
  fog_density = 0.32f;

  // zero out all object counters
  cylinder_array_cnt = 0;
  cylinder_array_color_cnt = 0;
  ring_array_color_cnt = 0;
  sphere_array_cnt = 0;
  sphere_array_color_cnt = 0;
  tricolor_cnt = 0;
  trimesh_c4u_n3b_v3f_cnt = 0;
  trimesh_n3b_v3f_cnt = 0;
  trimesh_n3f_v3f_cnt = 0;
}
        
/// destructor
OptiXRenderer::~OptiXRenderer(void) {
  if (context_created)
    destroy_context(); 
  wkf_timer_destroy(ort_timer);
}


//
// This routine enumerates the set of GPUs that are usable by OptiX,
// both in terms of their compatibility with the OptiX library we have
// compiled against, and also in terms of user preferences to exclude
// particular GPUs, GPUs that have displays attached, and so on.
//
unsigned int OptiXRenderer::device_list(int **devlist, char ***devnames) {
  unsigned int count=0;
  if (rtDeviceGetDeviceCount(&count) != RT_SUCCESS) {
    count = 0;
    if (devlist != NULL)
      *devlist = NULL;
    if (devnames != NULL)
      *devnames = NULL;
    return 0;
  }

  // check to see if the user wants to limit what device(s) are used
  unsigned int gpumask = 0xffffffff;
  const char *gpumaskstr = getenv("VMDOPTIXDEVICEMASK");
  if (gpumaskstr != NULL) {
    unsigned int tmp;
    if (sscanf(gpumaskstr, "%x", &tmp) == 1) {
      gpumask = tmp;
      msgInfo << "Using OptiX device mask '"
              << gpumaskstr << "'" << sendmsg;
    } else {
      msgInfo << "Failed to parse OptiX GPU device mask string '"
              << gpumaskstr << "'" << sendmsg;
    }
  }

  if (devlist != NULL) {
    *devlist = NULL;
    if (count > 0) {
      *devlist = (int *) calloc(1, count * sizeof(int));  
    }
  }
  if (devnames != NULL) {
    *devnames = NULL;
    if (count > 0) {
      *devnames = (char **) calloc(1, count * sizeof(char *));  
    }
  }

  // walk through the list of available devices and screen out
  // any that may cause problems with the version of OptiX we are using
  unsigned int i, goodcount;
  for (goodcount=0,i=0; i<count; i++) {
    // check user-defined GPU device mask for OptiX...
    if (!(gpumask & (1 << i))) {
//      printf("  Excluded GPU[%d] due to user-specified device mask\n", i);
      continue;
    } 

    // check user-requested exclusion of devices with display timeouts enabled
    int timeoutenabled;
    rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, 
                         sizeof(int), &timeoutenabled);
    if (timeoutenabled && getenv("VMDOPTIXNODISPLAYGPUS")) {
//      printf("  Excluded GPU[%d] due to user-specified display timeout exclusion \n", i);
      continue;
    } 

    //
    // screen for viable compute capability for this version of OptiX
    //
    // XXX this should be unnecessary with OptiX 3.6.x and later (I hope)
    //
    int compute_capability[2];
    rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, 
                         sizeof(compute_capability), compute_capability);
//    printf("OptiX GPU[%d] compute capability %d\n", i, compute_capability[0]);
#if OPTIX_VERSION <= 3051
    // exclude Maxwell and later GPUs if we're running OptiX 3.5.1 or earlier
    if (compute_capability[0] > 3) {
//      printf("  Excluded GPU[%d] due to unsupported compute capability\n", i);
      continue;
    }
#endif
#if OPTIX_VERSION <= 2051
    // exclude Kepler and later GPUs if we're running OptiX 2.5.1 or earlier
    if (compute_capability[0] > 2) {
//      printf("  Excluded GPU[%d] due to unsupported compute capability\n", i);
      continue;
    }
#endif

    // record all usable GPUs we find...
//    printf("Found usable GPU[%i]\n", i);
    if (devlist != NULL) {
//      printf("  Adding usable GPU[%i] to list[%d]\n", i, goodcount);
      (*devlist)[goodcount] = i;
    }
    if (devnames != NULL) {
      char *namebuf = (char *) calloc(1, 65 * sizeof(char));
      rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, 
                           64*sizeof(char), namebuf);
//      printf("  Adding usable GPU[%i] to list[%d]: '%s'\n", i, goodcount, namebuf);
      (*devnames)[goodcount] = namebuf;
    }
    goodcount++;
  }

  return goodcount;
}


unsigned int OptiXRenderer::device_count(void) {
#if 1
  return device_list(NULL, NULL);
#else
  unsigned int count=0;
  if (rtDeviceGetDeviceCount(&count) != RT_SUCCESS)
    count = 0;
  return count;
#endif
}


unsigned int OptiXRenderer::optix_version(void) {
  unsigned int version=0;
  if (rtGetVersion(&version) != RT_SUCCESS)
    version = 0;
  return version;
}


void OptiXRenderer::setup_context(int w, int h) {
  lasterror = RT_SUCCESS; // clear any error state
  width = w;
  height = h;

  char *verbstr = getenv("VMDOPTIXVERBOSE");
  if (verbstr != NULL) {
    printf("OptiXRenderer) verbosity config request: '%s'\n", verbstr);
    if (!strupcmp(verbstr, "MIN")) {
      verbose = RT_VERB_MIN;
      printf("OptiXRenderer) verbose setting: minimum\n");
    } else if (!strupcmp(verbstr, "TIMING")) {
      verbose = RT_VERB_TIMING;
      printf("OptiXRenderer) verbose setting: timing data\n");
    } else if (!strupcmp(verbstr, "DEBUG")) {
      verbose = RT_VERB_DEBUG;
      printf("OptiXRenderer) verbose setting: full debugging data\n");
    }
  }

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating context...\n");

  // allow runtime override of the default shader path for testing
  if (getenv("VMDOPTIXSHADERPATH")) {
    strcpy(shaderpath, getenv("VMDOPTIXSHADERPATH"));
    if (verbose == RT_VERB_DEBUG) 
      printf("OptiXRenderer) user-override shaderpath: '%s'\n", shaderpath);
  }

  // Create our objects and set state
  if (rtContextCreate(&context) != RT_SUCCESS) {
    msgErr << "OptiX: Failed to create OptiX rendering context" << sendmsg;
  }
  context_created = 1;

  // screen and set what GPU device(s) are used for this context
  // We shouldn't need the compuate capability exclusions
  // any more, starting with OptiX 3.6.x, but this will benefit from
  // other updates.
#if OPTIX_VERSION <= 3051
  int *optixdevlist;
  int optixdevcount = device_list(&optixdevlist, NULL);
  if (optixdevcount > 0) {
    RTERR( rtContextSetDevices(context, optixdevcount, optixdevlist) );
  }
#else
  if (getenv("VMDOPTIXDEVICE") != NULL) {
    int optixdev = atoi(getenv("VMDOPTIXDEVICE"));
    msgInfo << "Setting OptiX GPU device to: " << optixdev << sendmsg;
    RTERR( rtContextSetDevices(context, 1, &optixdev) );
  }
#endif
 
  // register ray types for both shadow and radiance rays
  RTERR( rtContextSetRayTypeCount(context, RT_RAY_TYPE_COUNT) );

  RTvariable max_depth;
  RTERR( rtContextDeclareVariable(context, "max_depth", &max_depth) );
  RTERR( rtVariableSet1i(max_depth, 20u) );
  if (getenv("VMDOPTIXMAXDEPTH")) {
    int maxdepth = atoi(getenv("VMDOPTIXMAXDEPTH"));
    if (maxdepth > 0 && maxdepth <= 20) {
      printf("OptiX: Setting maxdepth to %d...\n", maxdepth);
      RTERR( rtVariableSet1i(max_depth, maxdepth) );
    } else {
      printf("OptiX: ignoring out-of-range maxdepth to %d...\n", maxdepth);
    }
  }

  RTvariable radiance_ray_type;
  RTERR( rtContextDeclareVariable(context, "radiance_ray_type", &radiance_ray_type) );
  RTERR( rtVariableSet1ui(radiance_ray_type, 0u) );

  RTvariable shadow_ray_type;
  RTERR( rtContextDeclareVariable(context, "shadow_ray_type", &shadow_ray_type) );
  RTERR( rtVariableSet1ui(shadow_ray_type, 1u) );

  RTvariable epsilon;
  RTERR( rtContextDeclareVariable(context, "scene_epsilon", &epsilon) );

  float scene_epsilon = 5.e-5f;
  if (getenv("VMDOPTIXSCENEEPSILON") != NULL) {
    scene_epsilon = atof(getenv("VMDOPTIXSCENEEPSILON"));
    printf("OptiX: user override of scene epsilon: %g\n", scene_epsilon);
  }
  RTERR( rtVariableSet1f(epsilon, scene_epsilon) );

  allocate_framebuffer(width, height);

  // cylinder array programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "cylinder_array_bounds", &cylinder_array_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "cylinder_array_intersect", &cylinder_array_isct_pgm) );

  // color-per-cylinder array programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "cylinder_array_color_bounds", &cylinder_array_color_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "cylinder_array_color_intersect", &cylinder_array_color_isct_pgm) );

  // color-per-ring array programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "ring_array_color_bounds", &ring_array_color_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "ring_array_color_intersect", &ring_array_color_isct_pgm) );

  // sphere array programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "sphere_array_bounds", &sphere_array_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "sphere_array_intersect", &sphere_array_isct_pgm) );

  // color-per-sphere array programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "sphere_array_color_bounds", &sphere_array_color_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "sphere_array_color_intersect", &sphere_array_color_isct_pgm) );

  // tricolor list programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "tricolor_bounds", &tricolor_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "tricolor_intersect", &tricolor_isct_pgm) );

  // c4u_n3b_v3f
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "trimesh_c4u_n3b_v3f_bounds", &trimesh_c4u_n3b_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "trimesh_c4u_n3b_v3f_intersect", &trimesh_c4u_n3b_v3f_isct_pgm) );

  // n3f_v3f
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "trimesh_n3f_v3f_bounds", &trimesh_n3f_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "trimesh_n3f_v3f_intersect", &trimesh_n3f_v3f_isct_pgm) );

  // n3b_v3f
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "trimesh_n3b_v3f_bounds", &trimesh_n3b_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "trimesh_n3b_v3f_intersect", &trimesh_n3b_v3f_isct_pgm) );

#if !defined(VMDOPTIX_VCA)
  // set light variable now, populate the light buffer at render time...
  RTERR( rtContextDeclareVariable(context, "lights", &lightbuffer_v) );
#endif

  // Current progressive subframe index, used as part of generating
  // AA and AO random number sequences, particularly for runs
  // on network-connected VCA arrays where the use of RNG state buffers
  // that are saved/restored isn't workable.
#ifndef VMDOPTIX_VCA
  RTERR( rtContextDeclareVariable(context, "progressiveSubframeIndex", &progressive_index_v) );
  RTERR( rtVariableSet1ui(progressive_index_v, 0) );
#endif

  // set AO direct lighting scale factor
  RTERR( rtContextDeclareVariable(context, "ao_direct", &ao_direct_v) );
  RTERR( rtContextDeclareVariable(context, "ao_ambient", &ao_ambient_v) );

  // shadows, antialiasing, ambient occlusion
  RTERR( rtContextDeclareVariable(context, "shadows_enabled", &shadows_enabled_v) );
  RTERR( rtContextDeclareVariable(context, "aa_samples", &aa_samples_v) );
  RTERR( rtContextDeclareVariable(context, "ao_samples", &ao_samples_v) );

  // background color / gradient
  RTERR( rtContextDeclareVariable(context, "scene_bg_color", &scene_bg_color_v) );
  RTERR( rtContextDeclareVariable(context, "scene_bg_color_grad_top", &scene_bg_grad_top_v) );
  RTERR( rtContextDeclareVariable(context, "scene_bg_color_grad_bot", &scene_bg_grad_bot_v) );
  RTERR( rtContextDeclareVariable(context, "scene_gradient", &scene_gradient_v) );
  RTERR( rtContextDeclareVariable(context, "scene_gradient_topval", &scene_gradient_topval_v) );
  RTERR( rtContextDeclareVariable(context, "scene_gradient_botval", &scene_gradient_botval_v) );
  RTERR( rtContextDeclareVariable(context, "scene_gradient_invrange", &scene_gradient_invrange_v) );

  // cueing/fog variables
  RTERR( rtContextDeclareVariable(context, "fog_mode", &fog_mode_v) );
  RTERR( rtContextDeclareVariable(context, "fog_start", &fog_start_v) );
  RTERR( rtContextDeclareVariable(context, "fog_end", &fog_end_v) );
  RTERR( rtContextDeclareVariable(context, "fog_density", &fog_density_v) );

  // program for clearing the RNG buffers
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "clear_rng_buffers", &clear_rng_buffers_pgm) );

  // program for clearing the accumulation buffer
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "clear_accumulation_buffer", &clear_accumulation_buffer_pgm) );

  // program for clearing the framebuffer
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "clear_framebuffer", &clear_framebuffer_pgm) );

  // program for copying the accumulation buffer to the framebuffer
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "draw_accumulation_buffer", &draw_accumulation_buffer_pgm) );
}




void OptiXRenderer::update_rendering_state(void) {
  wkf_timer_start(ort_timer);

  long totaltris = tricolor_cnt + trimesh_c4u_n3b_v3f_cnt + 
                   trimesh_n3b_v3f_cnt + trimesh_n3f_v3f_cnt;

  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) object counts:\n");
    printf("OptiXRenderer)   %ld cylinders\n", cylinder_array_cnt + cylinder_array_color_cnt);
    printf("OptiXRenderer)   %ld rings\n", ring_array_color_cnt);
    printf("OptiXRenderer)   %ld spheres\n", sphere_array_cnt + sphere_array_color_cnt);
    printf("OptiXRenderer)   %ld triangles\n", totaltris);
    printf("OptiXRenderer) total objects: %ld\n",
           cylinder_array_cnt +  cylinder_array_color_cnt +
           ring_array_color_cnt + 
           sphere_array_cnt + sphere_array_color_cnt + 
           totaltris);

#if defined(ORT_USE_TEMPLATE_SHADERS)
    if (getenv("VMDOPTIXFORCEGENERALSHADER") == NULL) {
      printf("OptiXRenderer) using template-specialized shaders and materials:\n");
      int i;
      for (i=0; i<64; i++) {
        if (material_special_counts[i] > 0) {
          printf("OptiXRenderer) material_special[%d] usage count: %d\n", 
                 i, material_special_counts[i]); 
    
          printf("OptiXRenderer)   "
                 "Fog %s, "
                 "Shadows %s, "
                 "AO %s, "
                 "Outline %s, "
                 "Refl %s, "
                 "Trans %s\n",
                 onoffstr(i & 32),
                 onoffstr(i & 16),
                 onoffstr(i &  8),
                 onoffstr(i &  4),
                 onoffstr(i &  2),
                 onoffstr(i &  1));
        }
      }
      printf("OptiXRenderer)\n");
    } else {
      printf("OptiXRenderer) using fully general shader and materials.\n");
    }
#else
    printf("OptiXRenderer) using fully general shader and materials.\n");
#endif

  }

  int i;
  RTgeometrygroup geometrygroup;
  int instcount = geominstancelist.num();
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) instance objects: %d\n", instcount);

  RTERR( rtVariableSet3fv(scene_bg_color_v, scene_bg_color) );
  RTERR( rtVariableSet3fv(scene_bg_grad_top_v, scene_bg_grad_top) );
  RTERR( rtVariableSet3fv(scene_bg_grad_bot_v, scene_bg_grad_bot) );
  RTERR( rtVariableSet3fv(scene_gradient_v, scene_gradient) );
  RTERR( rtVariableSet1f(scene_gradient_topval_v, scene_gradient_topval) );
  RTERR( rtVariableSet1f(scene_gradient_botval_v, scene_gradient_botval) );

  if (verbose == RT_VERB_DEBUG) {
    printf("OptiX: scene bg mode: %d\n", scene_background_mode);

    printf("OptiX: scene bgsolid: %.2f %.2f %.2f\n", 
           scene_bg_color[0], scene_bg_color[1], scene_bg_color[2]);

    printf("OptiX: scene bggradT: %.2f %.2f %.2f\n", 
           scene_bg_grad_top[0], scene_bg_grad_top[1], scene_bg_grad_top[2]);

    printf("OptiX: scene bggradB: %.2f %.2f %.2f\n", 
           scene_bg_grad_bot[0], scene_bg_grad_bot[1], scene_bg_grad_bot[2]);
  
    printf("OptiX: bg gradient: %f %f %f  top: %f  bot: %f\n",
           scene_gradient[0], scene_gradient[1], scene_gradient[2],
           scene_gradient_topval, scene_gradient_botval);
  }

  // update in case the caller changed top/bottom values since last recalc
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);
  RTERR( rtVariableSet1f(scene_gradient_invrange_v, scene_gradient_invrange) );

  // Link up miss program depending on background rendering mode
  switch (scene_background_mode) {
    case RT_BACKGROUND_TEXTURE_SKY_SPHERE:
      RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "miss_gradient_bg_sky_sphere", &miss_pgm) );
      break;

    case RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE:
      RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "miss_gradient_bg_sky_plane", &miss_pgm) );
      break;

    case RT_BACKGROUND_TEXTURE_SOLID:
    default:
      RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "miss_solid_bg", &miss_pgm) );
      break;
  }

  RTERR( rtVariableSet1i(fog_mode_v, (int) fog_mode) );
  RTERR( rtVariableSet1f(fog_start_v, fog_start) );
  RTERR( rtVariableSet1f(fog_end_v, fog_end) );
  RTERR( rtVariableSet1f(fog_density_v, fog_density) );

  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) adding lights: %d\n", directional_lights.num());
  }

#if defined(VMDOPTIX_VCA)
  RTERR( rtContextDeclareVariable(context, "light_list", &light_list_v) );
  DirectionalLightList lights;
  memset( &lights, 0, sizeof( DirectionalLightList ) );
  lights.num_lights = directional_lights.num();
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy( (float*)( &lights.dirs[i] ), directional_lights[i].dir );
  }
  RTERR( rtVariableSetUserData( light_list_v, sizeof( DirectionalLightList ), &lights ) );
#else
  DirectionalLight *lbuf;
  RTERR( rtBufferCreate(context, RT_BUFFER_INPUT, &lightbuffer) );
  RTERR( rtBufferSetFormat(lightbuffer, RT_FORMAT_USER) );
  RTERR( rtBufferSetElementSize(lightbuffer, sizeof(DirectionalLight)) );
  RTERR( rtBufferSetSize1D(lightbuffer, directional_lights.num()) );
  RTERR( rtBufferMap(lightbuffer, (void **) &lbuf) );
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy((float*)&lbuf[i].dir, directional_lights[i].dir);
    vec_normalize((float*)&lbuf[i].dir);
  }
  RTERR( rtBufferUnmap(lightbuffer) );
  RTERR( rtVariableSetObject(lightbuffer_v, lightbuffer) );
#endif

  if (verbose == RT_VERB_DEBUG) printf("Finalizing OptiX scene graph...\n");

  // create group to hold instances
  RTERR( rtGeometryGroupCreate(context, &geometrygroup) );
  RTERR( rtGeometryGroupSetChildCount(geometrygroup, instcount) );
  for (i=0; i<instcount; i++) {
    RTERR( rtGeometryGroupSetChild(geometrygroup, i, geominstancelist[i]) );
  }

  // XXX we should create an acceleration object the instance shared
  //     by multiple PBC images

  // acceleration object for the geometrygroup
  RTERR( rtAccelerationCreate(context, &acceleration) );

  // Allow runtime override of acceleration builder and traverser
  // for performance testing/tuning
  const char *ort_builder   = getenv("VMDOPTIXBUILDER");
  const char *ort_traverser = getenv("VMDOPTIXTRAVERSER");
  if (ort_builder && ort_traverser) {
    RTERR( rtAccelerationSetBuilder(acceleration, ort_builder) );
    RTERR( rtAccelerationSetTraverser(acceleration, ort_traverser) );
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) user-override of AS: builder: '%s' traverser '%s'\n",
             ort_builder, ort_traverser);
    }
  } else if (ort_builder) {
    RTERR( rtAccelerationSetBuilder(acceleration, ort_builder) );
    RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) user-override of AS builder: '%s' (def traverser '%s')\n",
             ort_builder, "Bvh");
    }
  } else {
#if (OPTIX_VERSION >= 3050) && (OPTIX_VERSION < 3060) || (OPTIX_VERSION >= 3063)
    // OptiX 3.5.0 is the first to include the new fast "Trbvh" AS builder
    // OptiX 3.6.3 fixed Trbvh bugs on huge models
    RTERR( rtAccelerationSetBuilder(acceleration, "Trbvh") );
    RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
#else
    // For older revs of OptiX, the "MedianBvh" AS builder gives the 
    // best compromise between builder speed and ray tracing speed.
//    RTERR( rtAccelerationSetBuilder(acceleration, "Sbvh") );
//    RTERR( rtAccelerationSetBuilder(acceleration, "Bvh") );
    RTERR( rtAccelerationSetBuilder(acceleration, "MedianBvh") );
    RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
#endif
  }
  RTERR( rtGeometryGroupSetAcceleration(geometrygroup, acceleration) );
  RTERR( rtAccelerationMarkDirty(acceleration) );

  // create the root node of the scene graph
  RTERR( rtGroupCreate(context, &root_group) );
  RTERR( rtGroupSetChildCount(root_group, 1) );
  RTERR( rtGroupSetChild(root_group, 0, geometrygroup) );
  RTERR( rtContextDeclareVariable(context, "root_object", &root_object) );
  RTERR( rtVariableSetObject(root_object, root_group) );
  RTERR( rtContextDeclareVariable(context, "root_shadower", &root_shadower) );
  RTERR( rtVariableSetObject(root_shadower, root_group) );

  // create an acceleration object for the entire scene graph
  RTERR( rtAccelerationCreate(context, &root_acceleration) );
  RTERR( rtAccelerationSetBuilder(root_acceleration,"NoAccel") );
  RTERR( rtAccelerationSetTraverser(root_acceleration,"NoAccel") );
  RTERR( rtGroupSetAcceleration(root_group, root_acceleration) );
  RTERR( rtAccelerationMarkDirty(root_acceleration) );


  // do final state variable updates before rendering begins
  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) cam zoom factor %f\n", cam_zoom);
    printf("OptiXRenderer) cam stereo eye separation  %f\n", cam_stereo_eyesep);
    printf("OptiXRenderer) cam stereo convergence distance %f\n", 
           cam_stereo_convergence_dist);
    printf("OptiXRenderer) cam DoF focal distance %f\n", cam_dof_focal_dist);
    printf("OptiXRenderer) cam DoF f/stop %f\n", cam_dof_fnumber);
  }

  // define all of the standard camera params
  RTERR( rtContextDeclareVariable(context, "cam_zoom", &cam_zoom_v) );
  RTERR( rtContextDeclareVariable(context, "cam_pos", &cam_pos_v) );
  RTERR( rtContextDeclareVariable(context, "cam_U", &cam_U_v) );
  RTERR( rtContextDeclareVariable(context, "cam_V", &cam_V_v) );
  RTERR( rtContextDeclareVariable(context, "cam_W", &cam_W_v) );

  RTERR( rtVariableSet1f(cam_zoom_v,  cam_zoom) );
  RTERR( rtVariableSet3f( cam_pos_v,  0.0f,  0.0f,  2.0f) );
  RTERR( rtVariableSet3f(   cam_U_v,  1.0f,  0.0f,  0.0f) );
  RTERR( rtVariableSet3f(   cam_V_v,  0.0f,  1.0f,  0.0f) );
  RTERR( rtVariableSet3f(   cam_W_v,  0.0f,  0.0f, -1.0f) );

  // define stereoscopic camera parameters
  RTERR( rtContextDeclareVariable(context, "cam_stereo_eyesep", &cam_stereo_eyesep_v) );
  RTERR( rtContextDeclareVariable(context, "cam_stereo_convergence_dist", &cam_stereo_convergence_dist_v) );
  RTERR( rtVariableSet1f(cam_stereo_eyesep_v, cam_stereo_eyesep) );
  RTERR( rtVariableSet1f(cam_stereo_convergence_dist_v, cam_stereo_convergence_dist) );

  // define camera DoF parameters
  RTERR( rtContextDeclareVariable(context, "cam_dof_focal_dist", &cam_dof_focal_dist_v) );
  RTERR( rtContextDeclareVariable(context, "cam_dof_aperture_rad", &cam_dof_aperture_rad_v) );
  RTERR( rtVariableSet1f(cam_dof_focal_dist_v, cam_dof_focal_dist) );
  RTERR( rtVariableSet1f(cam_dof_aperture_rad_v, cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber)) );

  // register perspective camera ray gen programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, 
         "vmd_camera_perspective", &ray_gen_pgm_perspective) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, 
         "vmd_camera_perspective_dof", &ray_gen_pgm_perspective_dof) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, 
         "vmd_camera_perspective_stereo", &ray_gen_pgm_perspective_stereo) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, 
         "vmd_camera_perspective_stereo_dof", &ray_gen_pgm_perspective_stereo_dof) );

  // register othographic camera ray gen programs
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath,
         "vmd_camera_orthographic", &ray_gen_pgm_orthographic) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath,
         "vmd_camera_orthographic_stereo", &ray_gen_pgm_orthographic_stereo) );

  // flag ray gen programs as having been registered
  ray_gen_pgms_registered = 1;

  // for batch mode rendering, we prefer correctness to speed, so 
  // we currently ignore USE_REVERSE_SHADOW_RAYS_DEFAULT except when
  // running interactively.  When the reverse ray optimizatoin is 100%
  // bulletproof, we will use it for batch rendering also.
  RTERR( rtVariableSet1i(shadows_enabled_v, 
                         (shadows_enabled) ? RT_SHADOWS_ON : RT_SHADOWS_OFF) );

  RTERR( rtVariableSet1i(ao_samples_v, ao_samples) );
  RTERR( rtVariableSet1f(ao_ambient_v, ao_ambient) );
  RTERR( rtVariableSet1f(ao_direct_v, ao_direct) );

  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) setting sample counts:  AA %d  AO %d\n", aa_samples, ao_samples);
    printf("OptiXRenderer) setting AO factors:  AOA %f  AOD %f\n", ao_ambient, ao_direct);
  }

  //
  // Handle AA samples either internally with loops internal to 
  // each ray launch point thread, or externally by iterating over
  // multiple launches, adding each sample to an accumulation buffer,
  // or a hybrid combination of the two.  The final framebuffer output
  // is written by launching a special accumulation buffer drawing 
  // program that range clamps and converts the pixel data while copying
  // the GPU-local accumulation buffer to the final output buffer...
  //
  ext_aa_loops = 1;
  if (ao_samples > 0 || (aa_samples > 4)) {
    // if we have too much work for a single-pass rendering, we need to 
    // break it up into multiple passes or we risk having kernel timeouts
    ext_aa_loops = 1 + aa_samples;
    RTERR( rtVariableSet1i(aa_samples_v, 1) );
  } else { 
    // if the scene is simple, e.g. no AO rays and AA sample count is small,
    // we can run it in a single pass and get better performance
    RTERR( rtVariableSet1i(aa_samples_v, aa_samples + 1) );
  }
  RTERR( rtContextDeclareVariable(context, "accumulation_normalization_factor", &accum_norm_v) );
  RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(1 + aa_samples)) );

  if (verbose == RT_VERB_DEBUG) {
    if (ext_aa_loops > 1)
      printf("Running OptiX multi-pass: %d loops\n", ext_aa_loops);
    else
      printf("Running OptiX single-pass: %d total samples\n", 1+aa_samples);
  }

  // set the ray generation program to the active camera code...
  RTERR( rtContextSetEntryPointCount(context, RT_RAY_GEN_COUNT) );
  RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_CLEAR_RNG_BUFFERS, clear_rng_buffers_pgm) );
  RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, clear_accumulation_buffer_pgm) );
  RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_CLEAR_FRAMEBUFFER, clear_framebuffer_pgm) );

  // set the active ray gen program based on the active projection mode
  if (camera_projection == RT_PERSPECTIVE) {
    if (dof_enabled) {
      RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_dof) );
    } else {
      RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective) );
    }
  } else if (camera_projection == RT_ORTHOGRAPHIC) {
    if (dof_enabled) {
      printf("OptiXRenderer) DoF not available in orthographic projections\n");
    }
    RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic) );
  } else {
    msgErr << "OptiX: Illegal projection mode! Rendering aborted." << sendmsg;
    return;
  }

  RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_COPY_FINISH, draw_accumulation_buffer_pgm) );
  RTERR( rtContextSetMissProgram(context, RT_RAY_TYPE_RADIANCE, miss_pgm) );

  // enable exception handling for all defined entry points
  RTprogram exception_pgm;
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "exception", &exception_pgm) );
  unsigned int epcnt=0;
  RTERR( rtContextGetEntryPointCount(context, &epcnt) );
  unsigned int epidx;
  for (epidx=0; epidx<epcnt; epidx++) { 
    RTERR( rtContextSetExceptionProgram(context, epidx, exception_pgm) );
  }

  // enable all exceptions for debugging if requested
  if (getenv("VMDOPTIXDEBUG")) {
    printf("OptiX: Enabling all OptiX exceptions\n");
    RTERR( rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1) );
  }

  // increase default OptiX stack size to prevent runtime failures
  RTsize ssz;
  rtContextGetStackSize(context, &ssz);
  if (verbose == RT_VERB_DEBUG) printf("OptiX: original stack size: %ld\n", ssz);

  // a decent default stack size is 7KB
  long newstacksize = 7 * 1024;

  // allow runtime user override of the OptiX stack size in 
  // case we need to render a truly massive scene
  if (getenv("VMDOPTIXSTACKSIZE")) {
    if (verbose == RT_VERB_DEBUG) printf("OptiX: user stack size override: %ld\n", ssz);
    newstacksize = atoi(getenv("VMDOPTIXSTACKSIZE"));
  }
  rtContextSetStackSize(context, newstacksize);
  rtContextGetStackSize(context, &ssz);
  if (verbose == RT_VERB_DEBUG) printf("OptiX: new stack size: %ld\n", ssz);
#if !defined(VMDOPTIX_VCA)
  rtContextSetPrintEnabled(context, 1);
  rtContextSetPrintBufferSize(context, 1*1024*1024); 
#endif

#if defined(VMD_ENABLE_OPTIX_TIMEOUTS)
  // Add a custom OptiX timeout callback to see if we can overcome
  // some of the timeout issues we've had previously
  double timeoutlimit = 0.5;
  const char *tmstr = getenv("VMDOPTIXTIMEOUTLIMIT");
  if (tmstr) {
    timeoutlimit = atof(tmstr);
    printf("Setting OptiX timeout: %f sec\n", timeoutlimit);
  }

  if (verbose == RT_VERB_DEBUG)
    printf("Setting OptiX timeout: %f sec\n", timeoutlimit);
  
  RTERR( rtContextSetTimeoutCallback(context, vmd_timeout_cb, timeoutlimit) );
#endif
}


void OptiXRenderer::allocate_framebuffer(int fbwidth, int fbheight) {
  buffers_allocated = 1;

  // create intermediate GPU-local accumulation buffer
  RTERR( rtContextDeclareVariable(context, "accumulation_buffer", &accumulation_buffer_v) );
#ifdef VMDOPTIX_VCA
  RTERR( rtBufferCreate(context, RT_BUFFER_OUTPUT, &accumulation_buffer) );
  RTERR( rtBufferSetFormat(accumulation_buffer, RT_FORMAT_FLOAT4) );
  RTERR( rtBufferSetSize2D(accumulation_buffer, fbwidth, fbheight) );
#else
  RTERR( rtBufferCreate(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, &accumulation_buffer) );
  RTERR( rtBufferSetFormat(accumulation_buffer, RT_FORMAT_FLOAT4) );
  RTERR( rtBufferSetSize2D(accumulation_buffer, fbwidth, fbheight) );
#endif
  RTERR( rtVariableSetObject(accumulation_buffer_v, accumulation_buffer) );

#ifdef VMDOPTIX_VCA
  //RTERR( rtBufferCreate( context, RT_BUFFER_PROGRESSIVE_STREAM | RT_BUFFER_ENCODING_H264, &framebuffer ) );
  RTERR( rtBufferCreate( context, RT_BUFFER_PROGRESSIVE_STREAM, &framebuffer ) );
  RTERR( rtBufferSetFormat(framebuffer, RT_FORMAT_UNSIGNED_BYTE4) );
  RTERR( rtBufferSetSize2D(framebuffer, fbwidth, fbheight) );
  RTERR( rtBufferBindProgressiveStream( framebuffer, accumulation_buffer) );
#else
  // create output framebuffer
  RTERR( rtContextDeclareVariable(context, "framebuffer", &framebuffer_v) );
  RTERR( rtBufferCreate(context, RT_BUFFER_OUTPUT, &framebuffer) );
  RTERR( rtBufferSetFormat(framebuffer, RT_FORMAT_UNSIGNED_BYTE4) );
  RTERR( rtBufferSetSize2D(framebuffer, fbwidth, fbheight) );
  RTERR( rtVariableSetObject(framebuffer_v, framebuffer) );
#endif


#ifndef VMDOPTIX_VCA
  // create intermediate GPU-local RNG state buffer
  RTERR( rtContextDeclareVariable(context, "aa_rng_buffer", &aa_rng_buffer_v) );
  RTERR( rtBufferCreate(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, &aa_rng_buffer) );
  RTERR( rtBufferSetFormat(aa_rng_buffer, RT_FORMAT_UNSIGNED_INT) );
  RTERR( rtBufferSetSize2D(aa_rng_buffer, fbwidth, fbheight) );
  RTERR( rtVariableSetObject(aa_rng_buffer_v, aa_rng_buffer) );

  // create intermediate GPU-local RNG state buffer
  RTERR( rtContextDeclareVariable(context, "ao_rng_buffer", &ao_rng_buffer_v) );
  RTERR( rtBufferCreate(context, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, &ao_rng_buffer) );
  RTERR( rtBufferSetFormat(ao_rng_buffer, RT_FORMAT_UNSIGNED_INT) );
  RTERR( rtBufferSetSize2D(ao_rng_buffer, fbwidth, fbheight) );
  RTERR( rtVariableSetObject(ao_rng_buffer_v, ao_rng_buffer) );
#endif
}

void OptiXRenderer::resize_framebuffer(int fbwidth, int fbheight) {
#ifdef VMDOPTIX_VCA
  RTERR( rtBufferSetSize2D(framebuffer, width, height) );
  RTERR( rtBufferSetSize2D(accumulation_buffer, width, height) );
#else
  RTERR( rtBufferSetSize2D(framebuffer, width, height) );
  RTERR( rtBufferSetSize2D(accumulation_buffer, width, height) );
  RTERR( rtBufferSetSize2D(aa_rng_buffer, fbwidth, fbheight) );
  RTERR( rtBufferSetSize2D(ao_rng_buffer, fbwidth, fbheight) );
#endif
}


void OptiXRenderer::destroy_framebuffer() {
#ifndef VMDOPTIX_VCA
  RTERR( rtContextRemoveVariable(context, ao_rng_buffer_v) );
  RTERR( rtBufferDestroy(ao_rng_buffer) );
  RTERR( rtContextRemoveVariable(context, aa_rng_buffer_v) );
  RTERR( rtBufferDestroy(aa_rng_buffer) );
#endif

  RTERR( rtContextRemoveVariable(context, accumulation_buffer_v) );
  RTERR( rtBufferDestroy(accumulation_buffer) );
#ifndef VMDOPTIX_VCA
  RTERR( rtContextRemoveVariable(context, framebuffer_v) );
#endif
  RTERR( rtBufferDestroy(framebuffer) );
}


void OptiXRenderer::render_compile_and_validate(void) {
  //
  // finalize context validation, compilation, and AS generation 
  //
  double startctxtime = wkf_timer_timenow(ort_timer);

  if (verbose == RT_VERB_DEBUG) printf("Finalizing OptiX rendering kernels...\n");
  RTERR( rtContextValidate(context) );
  if (lasterror != RT_SUCCESS) {
    printf("OptiXRenderer) An error occured validating the context. Rendering is aborted.\n");
    return;
  }

  RTERR( rtContextCompile(context) );
  if (lasterror != RT_SUCCESS) {
    printf("OptiXRenderer) An error occured compiling the context. Rendering is aborted.\n");
    return;
  }
  double contextinittime = wkf_timer_timenow(ort_timer);
//  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) context validation and compilation time: %f secs\n", contextinittime - startctxtime);
  }

  //
  // Force OptiX to build the acceleration structure _now_ by using 
  // an empty launch.  This is done in OptiX sample 6...
  //
// #define ORT_RETRY_FAILED_AS_BUILD 1
#if defined(ORT_RETRY_FAILED_AS_BUILD)
  RTresult rc;
  rc = rtContextLaunch2D(context, RT_RAY_GEN_ACCUMULATE, 0, 0);
  RTERR( rc );
#else
  RTERR( rtContextLaunch2D(context, RT_RAY_GEN_ACCUMULATE, 0, 0) );
#endif
  double accelbuildtime = wkf_timer_timenow(ort_timer) - contextinittime;
//  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) acceleration structure build time: %f secs\n", accelbuildtime);
    printf("OptiXRenderer) launching OptiX rendering: %d x %d\n", width, height);
  }

#if defined(ORT_RETRY_FAILED_AS_BUILD)
  if (rc == RT_ERROR_MEMORY_ALLOCATION_FAILED) {
    const char *curbuilderstr = NULL;
    RTERR( rtAccelerationGetBuilder(acceleration, &curbuilderstr) );
    printf("Current OptiX builder str: '%s'\n", curbuilderstr);
    if (!strcmp(curbuilderstr, "Trbvh")) {
      // clear previous error so we don't abort immediately...
      lasterror = RT_SUCCESS;

      // issue warning, and try to rebuild the AS using a different builder
      printf("OptiXRenderer) Trbvh AS ran out of GPU memory, retrying with MedianBvh...\n");
      RTERR( rtAccelerationSetBuilder(acceleration, "MedianBvh") );
      RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );

      // try re-validating and re-compiling context after changin the
      // AS builder to something that can survive GPU memory shortages
      render_compile_and_validate(); 
    }
  }
#endif
}


#if defined(VMDOPTIX_INTERACTIVE_OPENGL)

static void *createoptixwindow(const char *wintitle, int width, int height) {
  printf("OptiXRenderer) Creating OptiX window: %d x %d...\n", width, height);

  void *win = glwin_create(wintitle, width, height);
  while (glwin_handle_events(win, GLWIN_EV_POLL_NONBLOCK) != 0);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glViewport(0, 0, width, height);
  glClear(GL_COLOR_BUFFER_BIT);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

  glwin_swap_buffers(win);

  return win;
}


static void drawoptiximage(void *win, RTbuffer buffer, int stereoon) {
  RTsize buffer_width, buffer_height;
  rtBufferGetSize2D(buffer, &buffer_width, &buffer_height);

  void * img;
  rtBufferMap(buffer, &img);

  int wsx, wsy;
  glwin_get_winsize(win, &wsx, &wsy);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glViewport(0, 0, wsx, wsy);
  glClear(GL_COLOR_BUFFER_BIT);

  glShadeModel(GL_FLAT);
  glViewport(0, 0, wsx, wsy);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, wsx, 0.0, wsy, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelZoom(1.0, 1.0); 

  if (stereoon) {
    //printf("wsz: %dx%d  bsz: %dx%d\n", wsx, wsy, buffer_width, buffer_height);
    unsigned char *leftimg = (unsigned char *) img;
    unsigned char *rightimg = leftimg + ((buffer_width * (buffer_height/2)) * 4);

    glDrawBuffer(GL_BACK_LEFT);
    glRasterPos2i(0, 0);
//    glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE); // anaglyph or testing
    glDrawPixels(buffer_width, buffer_height/2, GL_RGBA, GL_UNSIGNED_BYTE, leftimg);

    glDrawBuffer(GL_BACK_RIGHT);
    glRasterPos2i(0, 0);
//    glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE); // anaglyph or testing
    glDrawPixels(buffer_width, buffer_height/2, GL_RGBA, GL_UNSIGNED_BYTE, rightimg);
  } else {
    glRasterPos2i(0, 0);
    glDrawPixels(buffer_width, buffer_height, GL_RGBA, GL_UNSIGNED_BYTE, img);
  }

  rtBufferUnmap(buffer);

  glwin_swap_buffers(win);
}


static void print_ctx_devices(RTcontext ctx) {
  unsigned int devcount = 0;
  rtContextGetDeviceCount(ctx, &devcount);
  if (devcount > 0) {
    int *devlist = (int *) calloc(1, devcount * sizeof(int));
    rtContextGetDevices(ctx, devlist);
    printf("OptiXRenderer) Using %d device%s:\n", 
           devcount, (devcount == 1) ? "" : "s");

    unsigned int d;
    for (d=0; d<devcount; d++) {
      char devname[20];
      int cudadev=-1, kto=-1;
      RTsize totalmem;
      memset(devname, 0, sizeof(devname));

      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_NAME, sizeof(devname), devname);
      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(int), &kto);
      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(int), &cudadev);
      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(totalmem), &totalmem);

      printf("OptiXRenderer) [%u] %-16s  CUDA[%d], %.1fGB RAM", 
             d, devname, cudadev, totalmem / (1024.0*1024.0*1024.0));
      if (kto) {
        printf(", KTO");
      }
      printf("\n");
    }
    printf("OptiXRenderer)\n");

    free(devlist); 
  }
}


static void interactive_viewer_usage(RTcontext ctx, void *win) {
  printf("OptiXRenderer) VMD TachyonL-OptiX Interactive Ray Tracer help:\n");
  printf("OptiXRenderer) ===============================================\n");

  print_ctx_devices(ctx);

  // check for Spaceball/SpaceNavigator/Magellan input devices
  int havespaceball = ((glwin_spaceball_available(win)) && (getenv("VMDDISABLESPACEBALLXDRV") == NULL));
  printf("OptiXRenderer) Spaceball/SpaceNavigator/Magellan: %s\n",
         (havespaceball) ? "Available" : "Not available");

  // check for stereo-capable display
  int havestereo, havestencil;
  glwin_get_wininfo(win, &havestereo, &havestencil);
  printf("OptiXRenderer) Stereoscopic display: %s\n",
         (havestereo) ? "Available" : "Not available");

  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) General controls:\n");
  printf("OptiXRenderer)   space: save numbered snapshot image\n");
  printf("OptiXRenderer)       =: reset to initial view\n");
  printf("OptiXRenderer)       h: print this help info\n");
  printf("OptiXRenderer)       p: print current rendering parameters\n");
  printf("OptiXRenderer)   ESC,q: quit viewer\n");
  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) Display controls\n");
  printf("OptiXRenderer)      F1: override shadows on/off (off=AO off too)\n");
  printf("OptiXRenderer)      F2: override AO on/off\n");
  printf("OptiXRenderer)      F3: override DoF on/off\n");
  printf("OptiXRenderer)      F4: override Depth cueing on/off\n");
#ifdef USE_REVERSE_SHADOW_RAYS
  printf("OptiXRenderer)      F5: enable/disable shadow ray optimizations\n");
#endif
  printf("OptiXRenderer)     F12: toggle full-screen display on/off\n");
  printf("OptiXRenderer)   1-9,0: override samples per update auto-FPS off\n");
  printf("OptiXRenderer)      Up: increase DoF focal distance\n");
  printf("OptiXRenderer)    Down: decrease DoF focal distance\n");
  printf("OptiXRenderer)    Left: decrease DoF f/stop\n");
  printf("OptiXRenderer)   Right: increase DoF f/stop\n");
  printf("OptiXRenderer)       S: toggle stereoscopic display on/off (if avail)\n");
  printf("OptiXRenderer)       a: toggle AA/AO auto-FPS tuning on/off (on)\n");
  printf("OptiXRenderer)       g: toggle gradient sky xforms on/off (on)\n");
  printf("OptiXRenderer)       l: toggle light xforms on/off (on)\n");
  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) Mouse controls:\n");
  printf("OptiXRenderer)       f: mouse depth-of-field mode\n");
  printf("OptiXRenderer)       r: mouse rotation mode\n");
  printf("OptiXRenderer)       s: mouse scaling mode\n");
  printf("OptiXRenderer)       t: mouse translation mode\n");

  int movie_recording_enabled = (getenv("VMDOPTIXLIVEMOVIECAPTURE") != NULL);
  if (movie_recording_enabled) {
    printf("OptiXRenderer)\n");
    printf("OptiXRenderer) Movie recording controls:\n");
    printf("OptiXRenderer)       R: start/stop movie recording\n");
    printf("OptiXRenderer)       F: toggle movie FPS (24, 30, 60)\n");
  }
}


void OptiXRenderer::render_to_glwin(const char *filename) {
  enum RtMouseMode { RTMM_ROT=0, RTMM_TRANS=1, RTMM_SCALE=2, RTMM_DOF=3 };
  enum RtMouseDown { RTMD_NONE=0, RTMD_LEFT=1, RTMD_MIDDLE=2, RTMD_RIGHT=3 };
  RtMouseMode mm = RTMM_ROT;
  RtMouseDown mousedown = RTMD_NONE;
  int i;

  // flags to interactively enable/disable shadows, AO, DoF
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON_REVERSE : RT_SHADOWS_OFF;
#else
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
#endif

  int gl_fs_on=0; // fullscreen window state
  int owsx=0, owsy=0; // store last win size before fullscreen
  int gl_ao_on=(ao_samples > 0);
  int gl_dof_on, gl_dof_on_old;
  gl_dof_on=gl_dof_on_old=dof_enabled; 
  int gl_fog_on=(fog_mode != RT_FOG_NONE);

  // Enable live recording of a session to a stream of image files indexed
  // by their display presentation time, mapped to the nearest frame index
  // in a fixed-frame-rate image sequence (e.g. 24, 30, or 60 FPS), 
  // to allow subsequent encoding into a standard movie format.
  // XXX this feature is disabled by default at present, to prevent people
  //     from accidentally turning it on during a live demo or the like
  int movie_recording_enabled = (getenv("VMDOPTIXLIVEMOVIECAPTURE") != NULL);
  int movie_recording_on = 0;
  double movie_recording_start_time = 0.0;
  int movie_recording_fps = 30;
  const char *movie_recording_filebase = "vmdlivemovie.%05d.tga";
  if (getenv("VMDOPTIXLIVEMOVIECAPTUREFILEBASE"))
    movie_recording_filebase = getenv("VMDOPTIXLIVEMOVIECAPTUREFILEBASE");

  // Enable/disable Spaceball/SpaceNavigator/Magellan input 
  int spaceballenabled=(getenv("VMDDISABLESPACEBALLXDRV") == NULL) ? 1 : 0;

  // total AA/AO sample count
  int totalsamplecount=0;

  // counter for snapshots of live image...
  int snapshotcount=0;

  // flag to enable automatic AO sample count adjustment for FPS rate control
#if defined(VMDOPTIX_VCA)
  static int vcarunning=0;      // flag to indicate VCA streaming active/inactive
  int autosamplecount=0; // disable when targeting VCA for now
#else
  int autosamplecount=1;
#endif

  // flag to enable transformation of lights and gradient sky sphere, 
  // so that they track camera orientation as they do in the VMD OpenGL display
  int xformlights=1, xformgradientsphere=1;

  // prepare the majority of OptiX rendering state before we go into 
  // the interactive rendering loop
  update_rendering_state();
  render_compile_and_validate();

  // make a copy of state we're going to interactively manipulate,
  // so that we can recover to the original state on-demand
  int samples_per_pass = 1;
  int cur_aa_samples = aa_samples;
  int cur_ao_samples = ao_samples;
  float cam_zoom_orig = cam_zoom;
  float scene_gradient_orig[3] = {0.0f, 1.0f, 0.0f};
  vec_copy(scene_gradient_orig, scene_gradient);

  float cam_pos_orig[3] = {0.0f, 0.0f, 2.0f};
  float cam_U_orig[3] = {1.0f, 0.0f, 0.0f};
  float cam_V_orig[3] = {0.0f, 1.0f, 0.0f};
  float cam_W_orig[3] = {0.0f, 0.0f, -1.0f};
  float cam_pos[3], cam_U[3], cam_V[3], cam_W[3];
  vec_copy(cam_pos, cam_pos_orig);
  vec_copy(cam_U, cam_U_orig);
  vec_copy(cam_V, cam_V_orig);
  vec_copy(cam_W, cam_W_orig);

  // copy light directions
  DirectionalLight *cur_lights = (DirectionalLight *) calloc(1, directional_lights.num() * sizeof(DirectionalLight));
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy((float*)&cur_lights[i].dir, directional_lights[i].dir);
    vec_normalize((float*)&cur_lights[i].dir);
  }

  // create the display window
  void *win = createoptixwindow("VMD TachyonL-OptiX Interactive Ray Tracer", width, height);
  interactive_viewer_usage(context, win);
  
  // check for stereo-capable display
  int havestereo=0, havestencil=0;
  int stereoon=0, stereoon_old=0;
  glwin_get_wininfo(win, &havestereo, &havestencil);

  // Override AA/AO sample counts since we're doing progressive rendering.
  // Choosing an initial AO sample count of 1 will give us the peak progressive 
  // display update rate, but we end up wasting time on re-tracing many
  // primary rays.  The automatic FPS optimization scheme below will update
  // the number of samples per rendering pass and assign the best values for
  // AA/AO samples accordingly.
  cur_aa_samples = samples_per_pass;
  if (cur_ao_samples > 0) {
    cur_aa_samples = 1;
    cur_ao_samples = samples_per_pass;
  }

  const char *statestr = "|/-\\.";
  int done=0, winredraw=1, accum_count=0;
  int state=0, mousedownx=0, mousedowny=0;
  float cur_cam_zoom = cam_zoom_orig;
  double fpsexpave=0.0; 
  double oldtime = wkf_timer_timenow(ort_timer);
  int wsx=width, wsy=height;
  while (!done) { 
    int winevent=0;

    while ((winevent = glwin_handle_events(win, GLWIN_EV_POLL_NONBLOCK)) != 0) {
      int evdev, evval;
      char evkey;

      glwin_get_lastevent(win, &evdev, &evval, &evkey);
      glwin_get_winsize(win, &wsx, &wsy);

      if (evdev == GLWIN_EV_WINDOW_CLOSE) {
        printf("OptiXRenderer) display window closed, exiting...\n");
        done = 1;
        winredraw = 0;
      } else if (evdev == GLWIN_EV_KBD) {
        switch (evkey) {
          case  '1': autosamplecount=0; samples_per_pass=1; winredraw=1; break;
          case  '2': autosamplecount=0; samples_per_pass=2; winredraw=1; break;
          case  '3': autosamplecount=0; samples_per_pass=3; winredraw=1; break;
          case  '4': autosamplecount=0; samples_per_pass=4; winredraw=1; break;
          case  '5': autosamplecount=0; samples_per_pass=5; winredraw=1; break;
          case  '6': autosamplecount=0; samples_per_pass=6; winredraw=1; break;
          case  '7': autosamplecount=0; samples_per_pass=7; winredraw=1; break;
          case  '8': autosamplecount=0; samples_per_pass=8; winredraw=1; break;
          case  '9': autosamplecount=0; samples_per_pass=9; winredraw=1; break;
          case  '0': autosamplecount=0; samples_per_pass=10; winredraw=1; break;

          case  '=': /* recover back to initial state */
            vec_copy(scene_gradient, scene_gradient_orig);
            cam_zoom = cam_zoom_orig;
            vec_copy(cam_pos, cam_pos_orig);
            vec_copy(cam_U, cam_U_orig);
            vec_copy(cam_V, cam_V_orig);
            vec_copy(cam_W, cam_W_orig);

            // restore original light directions
            for (i=0; i<directional_lights.num(); i++) {
              vec_copy((float*)&cur_lights[i].dir, directional_lights[i].dir);
              vec_normalize((float*)&cur_lights[i].dir);
            }
            winredraw = 1;
            break;
 
          case  ' ': /* spacebar saves current image with counter */
            {
              char snapfilename[256];
              sprintf(snapfilename, "vmdsnapshot.%04d.tga", snapshotcount);
              if (OptiXWriteImage(snapfilename, framebuffer) != -1) {
                printf("OptiXRenderer) Saved snapshot to '%s'             \n",
                       snapfilename);
              }
              snapshotcount++; 
            }
            break;

          case  'a': /* toggle automatic sample count FPS tuning */
            autosamplecount = !(autosamplecount);
            printf("\nOptiXRenderer) Automatic AO sample count FPS tuning %s\n",
                   (autosamplecount) ? "enabled" : "disabled");
            break;

          case  'f': /* DoF mode */
            mm = RTMM_DOF;
            printf("\nOptiXRenderer) Mouse DoF aperture and focal dist. mode\n");
            break;

          case  'g': /* toggle gradient sky sphere xforms */
            xformgradientsphere = !(xformgradientsphere);
            printf("\nOptiXRenderer) Gradient sky sphere transformations %s\n",
                   (xformgradientsphere) ? "enabled" : "disabled");
            break;

          case  'h': /* print help message */
            printf("\n");
            interactive_viewer_usage(context, win);
            break;

          case  'l': /* toggle lighting xforms */
            xformlights = !(xformlights);
            printf("\nOptiXRenderer) Light transformations %s\n",
                   (xformlights) ? "enabled" : "disabled");
            break;

          case  'p': /* print current RT settings */
            printf("\nOptiXRenderer) Current Ray Tracing Parameters:\n"); 
            printf("OptiXRenderer) -------------------------------\n"); 
            printf("OptiXRenderer) Camera zoom: %f\n", cur_cam_zoom);
            printf("OptiXRenderer) Shadows: %s  Ambient occlusion: %s\n",
                   (gl_shadows_on) ? "on" : "off",
                   (gl_ao_on) ? "on" : "off");
            printf("OptiXRenderer) Antialiasing samples per-pass: %d\n",
                   cur_aa_samples);
            printf("OptiXRenderer) Ambient occlusion samples per-pass: %d\n",
                   cur_ao_samples);
            printf("OptiXRenderer) Depth-of-Field: %s f/num: %.1f  Foc. Dist: %.2f\n",
                   (gl_dof_on) ? "on" : "off", 
                   cam_dof_fnumber, cam_dof_focal_dist);
            printf("OptiXRenderer) Image size: %d x %d\n", width, height);
            break;

          case  'r': /* rotate mode */
            mm = RTMM_ROT;
            printf("\nOptiXRenderer) Mouse rotation mode\n");
            break;

          case  's': /* scaling mode */
            mm = RTMM_SCALE;
            printf("\nOptiXRenderer) Mouse scaling mode\n");
            break;

          case  'F': /* toggle live movie recording FPS (24, 30, 60) */
            if (movie_recording_enabled) {
              switch (movie_recording_fps) {
                case 24: movie_recording_fps = 30; break;
                case 30: movie_recording_fps = 60; break;
                case 60:
                default: movie_recording_fps = 24; break;
              }
              printf("\nOptiXRenderer) Movie recording FPS rate: %d\n", 
                     movie_recording_fps);
            } else {
              printf("\nOptiXRenderer) Movie recording not available.\n");
            }
            break;

          case  'R': /* toggle live movie recording mode on/off */
            if (movie_recording_enabled) {
              movie_recording_on = !(movie_recording_on);
              printf("\nOptiXRenderer) Movie recording %s\n",
                     (movie_recording_on) ? "STARTED" : "STOPPED");
              if (movie_recording_on)
                movie_recording_start_time = wkf_timer_timenow(ort_timer); 
            } else {
              printf("\nOptiXRenderer) Movie recording not available.\n");
            }
            break;

          case  'S': /* toggle stereoscopic display mode */
            if (havestereo) {
              stereoon = (!stereoon);
              printf("\nOptiXRenderer) Stereoscopic display %s\n",
                     (stereoon) ? "enabled" : "disabled");
              winredraw = 1;
            } else {
              printf("\nOptiXRenderer) Stereoscopic display unavailable\n");
            }
            break;
 
          case  't': /* translation mode */
            mm = RTMM_TRANS;
            printf("\nOptiXRenderer) Mouse translation mode\n");
            break;
            
          case  'q': /* 'q' key */
          case  'Q': /* 'Q' key */
          case 0x1b: /* ESC key */
            printf("\nOptiXRenderer) Exiting on user input.               \n");
            done=1; /* exit from interactive RT window */
            break;
        }
      } else if (evdev != GLWIN_EV_NONE) {
        switch (evdev) {
          case GLWIN_EV_KBD_F1: /* turn shadows on/off */
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
            gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON_REVERSE : RT_SHADOWS_OFF;
#else
            gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
            // gl_shadows_on = (!gl_shadows_on);
#endif

            printf("\n");
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
            printf("OptiXRenderer) Shadows %s\n",
                   (gl_shadows_on) ? "enabled (reversal opt.)" : "disabled");
#else
            printf("OptiXRenderer) Shadows %s\n",
                   (gl_shadows_on) ? "enabled" : "disabled");
#endif
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F2: /* turn AO on/off */
            gl_ao_on = (!gl_ao_on); 
            printf("\n");
            printf("OptiXRenderer) Ambient occlusion %s\n",
                   (gl_ao_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F3: /* turn DoF on/off */
            gl_dof_on = (!gl_dof_on);
            printf("\n");
            if ((camera_projection == RT_ORTHOGRAPHIC) && gl_dof_on) {
              gl_dof_on=0; 
              printf("OptiXRenderer) Depth-of-field not available in orthographic mode\n");
            }
            printf("OptiXRenderer) Depth-of-field %s\n",
                   (gl_dof_on) ? "enabled" : "disabled");
            winredraw = 1;
            break;

          case GLWIN_EV_KBD_F4: /* turn fog/depth cueing on/off */
            gl_fog_on = (!gl_fog_on); 
            printf("\n");
            printf("OptiXRenderer) Depth cueing %s\n",
                   (gl_fog_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

#ifdef USE_REVERSE_SHADOW_RAYS
          case GLWIN_EV_KBD_F5: /* toggle shadow ray reversal on/off */
            if (gl_shadows_on == RT_SHADOWS_ON) 
              gl_shadows_on = RT_SHADOWS_ON_REVERSE;
            else if (gl_shadows_on == RT_SHADOWS_ON_REVERSE)
              gl_shadows_on = RT_SHADOWS_ON;
            printf("\n");
            printf("OptiXRenderer) Shadow ray reversal %s\n",
                   (gl_shadows_on==RT_SHADOWS_ON_REVERSE) ? "enabled" : "disabled");
            winredraw = 1; 
            break;
#endif

          case GLWIN_EV_KBD_F12: /* toggle full-screen window on/off */
            gl_fs_on = (!gl_fs_on);
            printf("\nOptiXRenderer) Toggling fullscreen window %s\n",
                   (gl_fs_on) ? "on" : "off");
            if (gl_fs_on) { 
              if (glwin_fullscreen(win, gl_fs_on, 0) == 0) {
                owsx = wsx;
                owsy = wsy;
                glwin_get_winsize(win, &wsx, &wsy);
              } else {
                printf("OptiXRenderer) Fullscreen mode note available\n");
              }
            } else {
              glwin_fullscreen(win, gl_fs_on, 0);
              glwin_resize(win, owsx, owsy);
            }
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_UP: /* change depth-of-field focal dist */
            cam_dof_focal_dist *= 1.02f; 
            printf("\nOptiXRenderer) DoF focal dist: %f\n", cam_dof_focal_dist);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_DOWN: /* change depth-of-field focal dist */
            cam_dof_focal_dist *= 0.96f; 
            if (cam_dof_focal_dist < 0.02f) cam_dof_focal_dist = 0.02f;
            printf("\nOptiXRenderer) DoF focal dist: %f\n", cam_dof_focal_dist);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_RIGHT: /* change depth-of-field f/stop number */
            cam_dof_fnumber += 1.0f; 
            printf("\nOptiXRenderer) DoF f/stop: %f\n", cam_dof_fnumber);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_LEFT: /* change depth-of-field f/stop number */
            cam_dof_fnumber -= 1.0f; 
            if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
            printf("\nOptiXRenderer) DoF f/stop: %f\n", cam_dof_fnumber);
            winredraw = 1; 
            break;

          case GLWIN_EV_MOUSE_MOVE:
            if (mousedown != RTMD_NONE) {
              int x, y;
              glwin_get_mousepointer(win, &x, &y);

              float zoommod = 2.0f*cur_cam_zoom/cam_zoom_orig;
              float txdx = (x - mousedownx) * zoommod / wsx;
              float txdy = (y - mousedowny) * zoommod / wsy;
              if (mm != RTMM_SCALE) {
                mousedownx = x;
                mousedowny = y;
              }

              if (mm == RTMM_ROT) {
                Matrix4 rm;
                if (mousedown == RTMD_LEFT) {
                  // when zooming in further from the initial view, we
                  // rotate more slowly so control remains smooth
                  rm.rotate_axis(cam_V, -txdx);
                  rm.rotate_axis(cam_U, -txdy);
                } else if (mousedown == RTMD_MIDDLE || 
                           mousedown == RTMD_RIGHT) {
                  rm.rotate_axis(cam_W, txdx);
                }
                rm.multpoint3d(cam_pos, cam_pos);
                rm.multnorm3d(cam_U, cam_U);
                rm.multnorm3d(cam_V, cam_V);
                rm.multnorm3d(cam_W, cam_W);

                if (xformgradientsphere) {
                  rm.multnorm3d(scene_gradient, scene_gradient);
                }
 
                if (xformlights) {
                  // update light directions (comparatively costly)
                  for (i=0; i<directional_lights.num(); i++) {
                    rm.multnorm3d((float*)&cur_lights[i].dir, (float*)&cur_lights[i].dir);
                  }
                }

                winredraw = 1;
              } else if (mm == RTMM_TRANS) {
                if (mousedown == RTMD_LEFT) {
                  float dU[3], dV[3];
                  vec_scale(dU, -txdx, cam_U);
                  vec_scale(dV,  txdy, cam_V);
                  vec_add(cam_pos, cam_pos, dU); 
                  vec_add(cam_pos, cam_pos, dV); 
                } else if (mousedown == RTMD_MIDDLE || 
                           mousedown == RTMD_RIGHT) {
                  float dW[3];
                  vec_scale(dW, txdx, cam_W);
                  vec_add(cam_pos, cam_pos, dW); 
                } 
                winredraw = 1;
              } else if (mm == RTMM_SCALE) {
                float txdx = (x - mousedownx) * 2.0 / wsx;
                float zoominc = 1.0 - txdx;
                if (zoominc < 0.01) zoominc = 0.01;
                cam_zoom = cur_cam_zoom * zoominc;
                winredraw = 1;
              } else if (mm == RTMM_DOF) {
                cam_dof_fnumber += txdx * 20.0f;
                if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
                cam_dof_focal_dist += -txdy; 
                if (cam_dof_focal_dist < 0.01f) cam_dof_focal_dist = 0.01f;
                winredraw = 1;
              }
            }
            break;

          case GLWIN_EV_MOUSE_LEFT:
          case GLWIN_EV_MOUSE_MIDDLE:
          case GLWIN_EV_MOUSE_RIGHT:
            if (evval) {
              glwin_get_mousepointer(win, &mousedownx, &mousedowny);
              cur_cam_zoom = cam_zoom;

              if (evdev == GLWIN_EV_MOUSE_LEFT) mousedown = RTMD_LEFT;
              else if (evdev == GLWIN_EV_MOUSE_MIDDLE) mousedown = RTMD_MIDDLE;
              else if (evdev == GLWIN_EV_MOUSE_RIGHT) mousedown = RTMD_RIGHT;
            } else {
              mousedown = RTMD_NONE;
            }
            break;

          case GLWIN_EV_MOUSE_WHEELUP:
            cam_zoom /= 1.1f; winredraw = 1; break;

          case GLWIN_EV_MOUSE_WHEELDOWN:
            cam_zoom *= 1.1f; winredraw = 1; break;
        }
      }
    }


    //
    // Support for Spaceball/Spacenavigator/Magellan devices that use
    // X11 ClientMessage protocol....
    //
    if (spaceballenabled) {
      // Spaceball/Spacenavigator/Magellan event state variables
      int tx=0, ty=0, tz=0, rx=0, ry=0, rz=0, buttons=0;
      if (glwin_get_spaceball(win, &rx, &ry, &rz, &tx, &ty, &tz, &buttons)) {
        float zoommod = 2.0f*cam_zoom/cam_zoom_orig;
        float divlen = sqrtf(wsx*wsx + wsy*wsy) * 50;

        // check for rotation and handle it...
        if (rx != 0 || ry !=0 || rz !=0) {
          Matrix4 rm;
          rm.rotate_axis(cam_U, -rx * zoommod / divlen);
          rm.rotate_axis(cam_V, -ry * zoommod / divlen);
          rm.rotate_axis(cam_W, -rz * zoommod / divlen);

          rm.multpoint3d(cam_pos, cam_pos);
          rm.multnorm3d(cam_U, cam_U);
          rm.multnorm3d(cam_V, cam_V);
          rm.multnorm3d(cam_W, cam_W);

          if (xformgradientsphere) {
            rm.multnorm3d(scene_gradient, scene_gradient);
          }

          if (xformlights) {
            // update light directions (comparatively costly)
            for (i=0; i<directional_lights.num(); i++) {
              rm.multnorm3d((float*)&cur_lights[i].dir, (float*)&cur_lights[i].dir);
            }
          }
          winredraw = 1;
        }

        // check for translation and handle it...
        if (tx != 0 || ty !=0 || tz !=0) {
          float dU[3], dV[3], dW[3];
          vec_scale(dU, -tx * zoommod / divlen, cam_U);
          vec_scale(dV, -ty * zoommod / divlen, cam_V);
          vec_scale(dW, -tz * zoommod / divlen, cam_W);
          vec_add(cam_pos, cam_pos, dU);
          vec_add(cam_pos, cam_pos, dV);
          vec_add(cam_pos, cam_pos, dW);
          winredraw = 1;
        }

        // check for button presses to reset the view
        if (buttons & 1 || buttons & 2) {
          vec_copy(scene_gradient, scene_gradient_orig);
          cam_zoom = cam_zoom_orig;
          vec_copy(cam_pos, cam_pos_orig);
          vec_copy(cam_U, cam_U_orig);
          vec_copy(cam_V, cam_V_orig);
          vec_copy(cam_W, cam_W_orig);

          // restore original light directions
          for (i=0; i<directional_lights.num(); i++) {
            vec_copy((float*)&cur_lights[i].dir, directional_lights[i].dir);
            vec_normalize((float*)&cur_lights[i].dir);
          }
          winredraw = 1;
        }

      }
    }


    //
    // handle window resizing, stereoscopic mode changes,
    // destroy and recreate affected OptiX buffers
    //
    int resize_buffers=0;
    if (wsx != width) {
      width = wsx;
      resize_buffers=1;
    }
 
    if (wsy != height || (stereoon != stereoon_old)) {
      if (stereoon) {
        if (height != wsy * 2) {
          height = wsy * 2; 
          resize_buffers=1;
        }
      } else {
        height = wsy;
        resize_buffers=1;
      }
    }

#ifdef VMDOPTIX_VCA
    // 
    // Check for all conditions that would require modifying OptiX state
    // and tell the VCA to stop progressive rendering before we modify 
    // the rendering state, 
    //
    if (done || winredraw || resize_buffers ||
        (stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {
      // need to issue stop command before editing optix objects
      if (vcarunning) {
        rtContextStopProgressive( context );
        vcarunning=0;
      }
    }
#endif

    // check if stereo mode or DoF mode changed, both cases
    // require changing the active ray gen program
    if ((stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {

      // when stereo mode changes, we have to regenerate the
      // the RNG, accumulation buffer, and framebuffer
      if (stereoon != stereoon_old) {
        resize_buffers=1;
      }

      // update stereo and DoF state
      stereoon_old = stereoon;
      gl_dof_on_old = gl_dof_on;

      if (stereoon) {
        // set double-height over/under stereo display dimensions if needed
//        printf("\nStereo just enabled...\n");

        // set the active ray gen program based on the active projection mode
        if (camera_projection == RT_PERSPECTIVE) {
          if (gl_dof_on) {
            RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_stereo_dof) );
          } else {
            RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_stereo) );
          }
        } else {
          RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic_stereo) );
        }
      } else {
        // stereo was just turned off, switch ray gen programs
//        printf("\nStereo just disabled...\n");

        // set the active ray gen program based on the active projection mode
        if (camera_projection == RT_PERSPECTIVE) {
          if (gl_dof_on) {
            RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_dof) );
          } else {
            RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective) );
          }
        } else {
          RTERR( rtContextSetRayGenerationProgram(context, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic) );
        }
      }
    }

    if (resize_buffers) {
#if 1 || defined(VMDOPTIX_VCA)
      // resize existing buffers to avoid crash on VCA
      resize_framebuffer(width, height);
#else
      destroy_framebuffer();
      allocate_framebuffer(width, height);
#endif

      // when movie recording is enabled, print the window size as a guide
      // since the user might want to precisely control the size or 
      // aspect ratio for a particular movie format, e.g. 1080p, 4:3, 16:9
      if (movie_recording_enabled) {
        printf("\rOptiXRenderer) Window resize: %d x %d                               \n", width, height);
      }

      winredraw=1;
    }

    int frame_ready = 1; // Default to true for the non-VCA case
    unsigned int subframe_count = 1;
    if (!done) {
      //
      // If the user interacted with the window in a meaningful way, we
      // need to update the OptiX rendering state, recompile and re-validate
      // the context, and then re-render...
      //
      if (winredraw) {
        // update camera parameters
        RTERR( rtVariableSet1f( cam_zoom_v, cam_zoom) );
        RTERR( rtVariableSet3fv( cam_pos_v, cam_pos) );
        RTERR( rtVariableSet3fv(   cam_U_v, cam_U) );
        RTERR( rtVariableSet3fv(   cam_V_v, cam_V) );
        RTERR( rtVariableSet3fv(   cam_W_v, cam_W) );
        RTERR( rtVariableSet3fv(scene_gradient_v, scene_gradient) );
 
        // update shadow state 
        RTERR( rtVariableSet1i(shadows_enabled_v, gl_shadows_on) );

        // update depth cueing state
        RTERR( rtVariableSet1i(fog_mode_v, 
                 (int) (gl_fog_on) ? fog_mode : RT_FOG_NONE) );
 
        // update/recompute DoF values 
        RTERR( rtVariableSet1f(cam_dof_focal_dist_v, cam_dof_focal_dist) );
        RTERR( rtVariableSet1f(cam_dof_aperture_rad_v, cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber)) );

        // update light directions in the OptiX light buffer, but
        // only when xformlights is set, otherwise we take a 50%
        // speed hit when using a remote VCA cluster for rendering
        if (xformlights) {
#if defined(VMDOPTIX_VCA)
          DirectionalLightList lights;
          memset( &lights, 0, sizeof( DirectionalLightList ) );
          lights.num_lights = directional_lights.num();
          for (i=0; i<directional_lights.num(); i++) {
            //vec_copy( (float*)( &lights.dirs[i] ), cur_lights[i].dir );
            lights.dirs[i] = cur_lights[i].dir;
          }
          RTERR( rtVariableSetUserData( light_list_v, sizeof( DirectionalLightList ), &lights ) );
#else
          DirectionalLight *lbuf;
          RTERR( rtBufferMap(lightbuffer, (void **) &lbuf) );
          for (i=0; i<directional_lights.num(); i++) {
            vec_copy((float*)&lbuf[i].dir, (float*)&cur_lights[i].dir);
            vec_normalize((float*)&lbuf[i].dir);
          }
          RTERR( rtBufferUnmap(lightbuffer) );
#endif
        }

        // reset accumulation buffer 
        accum_count=0;
        totalsamplecount=0;


        // 
        // Sample count updates and OptiX state must always remain in 
        // sync, so if we only update sample count state during redraw events,
        // that's the only time we should recompute the sample counts, since
        // they also affect normalization factors for the accumulation buffer
        // in the non-VCA case.
        //

        // Update sample counts to achieve target interactivity
        if (autosamplecount) {
          if (fpsexpave > 37)
            samples_per_pass++;
          else if (fpsexpave < 30) 
            samples_per_pass--;
    
          // clamp sample counts to a "safe" range
          if (samples_per_pass > 14)
            samples_per_pass=14;
          if (samples_per_pass < 1)
            samples_per_pass=1;
        } 

        // split samples per pass either among AA and AO, depending on
        // whether DoF and AO are enabled or not. 
        if (gl_shadows_on && gl_ao_on) {
          if (gl_dof_on) {
            if (samples_per_pass < 4) {
              cur_aa_samples=samples_per_pass;
              cur_ao_samples=1;
            } else {
              int s = (int) sqrtf(samples_per_pass);
              cur_aa_samples=s;
              cur_ao_samples=s;
            }
          } else {
            cur_aa_samples=1;
            cur_ao_samples=samples_per_pass;
          }
        } else {
          cur_aa_samples=samples_per_pass;
          cur_ao_samples=0;
        }

        // update the current AA/AO sample counts since they may be changing if
        // FPS autotuning is enabled...
        RTERR( rtVariableSet1i(aa_samples_v, cur_aa_samples) );

        // observe latest AO enable/disable flag, and sample count
        if (gl_shadows_on && gl_ao_on) {
          RTERR( rtVariableSet1i(ao_samples_v, cur_ao_samples) );
        } else {
          cur_ao_samples = 0;
          RTERR( rtVariableSet1i(ao_samples_v, 0) );
        }

#ifdef VMDOPTIX_VCA
        RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(cur_aa_samples)) );
#endif
      } 


      //
      // The non-VCA code path must handle the accumulation buffer 
      // for itself, correctly rescaling the accumulated samples when
      // drawing to the output framebuffer.  
      //
      // The VCA code path takes care of normalization for itself.
      //
#ifndef VMDOPTIX_VCA
      // The accumulation buffer normalization factor must be updated
      // to reflect the total accumulation count before the accumulation
      // buffer is drawn to the output framebuffer
      RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(cur_aa_samples + accum_count)) );

      // The progressive rendering index must be updated to ensure that
      // the RNGs for AA and AO get correctly re-seeded
      RTERR( rtVariableSet1ui(progressive_index_v, accum_count) );

      // Force context compilation/validation
      // If no state has changed, there's no need to recompile/validate.
      // This call can be omitted since OptiX will do this automatically
      // at the next rtContextLaunchXXX() call.
//      render_compile_and_validate();
#endif


      //
      // run the renderer 
      //
      frame_ready = 1; // Default to true for the non-VCA case
      subframe_count = 1;
      if (lasterror == RT_SUCCESS) {
        if (winredraw) {
#ifdef VMDOPTIX_VCA
          // start the VCA doing progressive rendering...
          RTERR( rtContextLaunchProgressive2D(context, RT_RAY_GEN_ACCUMULATE, width, height, 0) );
          vcarunning=1;
#else
          RTERR( rtContextLaunch2D(context, RT_RAY_GEN_CLEAR_RNG_BUFFERS, width, height) );
          RTERR( rtContextLaunch2D(context, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, width, height) );
#endif
          winredraw=0;
        }

#ifdef VMDOPTIX_VCA
        // Wait for the next frame to arrive
        //RTERR( rtContextWaitForProgressiveUpdate( context, RT_PROGRESSIVE_WAIT_NEXT,
        //                                          &stream, &subframe_count, &max_subframes ) );
        RTERR( rtBufferGetProgressiveUpdateReady( framebuffer, &frame_ready, &subframe_count, 0 ) );
        totalsamplecount = subframe_count * samples_per_pass;
#else
        // iterate, adding to the accumulation buffer...
        RTERR( rtContextLaunch2D(context, RT_RAY_GEN_ACCUMULATE, width, height) );
        accum_count+=cur_aa_samples;
        totalsamplecount += samples_per_pass;

        // copy the accumulation buffer image data to the framebuffer and
        // perform type conversion and normaliztion on the image data...
        RTERR( rtContextLaunch2D(context, RT_RAY_GEN_COPY_FINISH, width, height) );
#endif

        if (lasterror == RT_SUCCESS) {
          if (frame_ready) {
            drawoptiximage(win, framebuffer, stereoon); // display output image

            // if live movie recording is on, we save every displayed frame
            // to a sequence sequence of image files, with each file numbered
            // by its frame index, which is computed by the multiplying image 
            // presentation time by the image sequence fixed-rate-FPS value.
            if (movie_recording_enabled && movie_recording_on) {
              double now = wkf_timer_timenow(ort_timer);
              double frametime = now - movie_recording_start_time;
              int frameindex = frametime * movie_recording_fps;
              char moviefilename[256];
              sprintf(moviefilename, movie_recording_filebase, frameindex);
              if (OptiXWriteImage(moviefilename, framebuffer) == -1) {
                movie_recording_on = 0;
                printf("\n");
                printf("OptiXRenderer) ERROR during writing image during movie recording!\n");
                printf("OptiXRenderer) Movie recording STOPPED\n");
              }              
            }
          }
        } else {
          printf("OptiXRenderer) An error occured during rendering. Rendering is aborted.\n");
          done=1;
          break;
        }
      } else {
        printf("OptiXRenderer) An error occured in AS generation. Rendering is aborted.\n");
        done=1;
        break;
      }
    }

    if (!done && frame_ready) {
      double newtime = wkf_timer_timenow(ort_timer);
      double frametime = (newtime-oldtime) + 0.00001f;
      oldtime=newtime;

      // compute exponential moving average for exp(-1/10)
      double framefps = 1.0f/frametime;
      fpsexpave = (fpsexpave * 0.90) + (framefps * 0.10);

#if 0 && defined(VMDOPTIX_VCA)
      printf("OptiXRenderer) %c AA:%2d AO:%2d, %4d total  FPS(ave): %.1f  %.4f s/frame subframes: %2d  \n",
             statestr[state], 
             cur_aa_samples, cur_ao_samples, totalsamplecount, 
             fpsexpave, frametime, subframe_count);
#elif 0 && defined(VMDOPTIX_VCA)
      printf("OptiXRenderer) %c AA:%2d AO:%2d, %4d subframes  FPS(ave): %.1f  %.4f s/frame   \r",
             statestr[state], cur_aa_samples, cur_ao_samples, subframe_count, 
             fpsexpave, frametime);
#else
      printf("OptiXRenderer) %c AA:%2d AO:%2d, %4d total  FPS(ave): %.1f  %.4f s/frame   \r",
             statestr[state], cur_aa_samples, cur_ao_samples, totalsamplecount, 
             fpsexpave, frametime);
#endif
      fflush(stdout);
      state = (state+1) & 3;
    }

  } // end of per-cycle event processing

  printf("\n");

  // write the output image upon exit...
  if (lasterror == RT_SUCCESS) {
    wkf_timer_start(ort_timer);
    // write output image
    OptiXWriteImage(filename, framebuffer);
    wkf_timer_stop(ort_timer);

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) image file I/O time: %f secs\n", wkf_timer_time(ort_timer));
    }
  }

  glwin_destroy(win);
}

#endif


void OptiXRenderer::render_to_file(const char *filename) {
  update_rendering_state();
  render_compile_and_validate();

  //
  // run the renderer 
  //
  if (lasterror == RT_SUCCESS) {
    wkf_timer_start(ort_timer);

    // initialize the RNG buffers
    RTERR( rtContextLaunch2D(context, RT_RAY_GEN_CLEAR_RNG_BUFFERS, width, height) );

    // clear the accumulation buffer
    RTERR( rtContextLaunch2D(context, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, width, height) );

    // fill the accumulation buffer with image data...
    int accum_sample;
    for (accum_sample=0; accum_sample<ext_aa_loops; accum_sample++) {
      // The progressive rendering index must be updated to ensure that
      // the RNGs for AA and AO get correctly re-seeded
      RTERR( rtVariableSet1ui(progressive_index_v, accum_sample) );

      RTERR( rtContextLaunch2D(context, RT_RAY_GEN_ACCUMULATE, width, height) );
    }

    // copy the accumulation buffer image data to the framebuffer and perform
    // type conversion and normaliztion on the image data...
    RTERR( rtContextLaunch2D(context, RT_RAY_GEN_COPY_FINISH, width, height) );
    wkf_timer_stop(ort_timer);

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) rendering time: %f secs\n", wkf_timer_time(ort_timer));
    }

    if (lasterror == RT_SUCCESS) {
      wkf_timer_start(ort_timer);
      // write output image
      OptiXWriteImage(filename, framebuffer);
      wkf_timer_stop(ort_timer);

      if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
        printf("OptiXRenderer) image file I/O time: %f secs\n", wkf_timer_time(ort_timer));
      }
    } else {
      printf("OptiXRenderer) An error during rendering, rendering is aborted.\n");
    }
  } else {
    printf("OptiXRenderer) An error occured in AS generation. Rendering is aborted.\n");
  }
}


void OptiXRenderer::destroy_context() {
#ifdef VMDOPTIX_VCA
  // ensure that there's no way we could be leaving the VCA running
  rtContextStopProgressive( context );
#endif

  if (buffers_allocated)
    destroy_framebuffer();
  
  if (ray_gen_pgms_registered) {
    RTERR( rtProgramDestroy(ray_gen_pgm_perspective) );
    RTERR( rtProgramDestroy(ray_gen_pgm_perspective_stereo) );
    RTERR( rtProgramDestroy(ray_gen_pgm_orthographic) );
    RTERR( rtProgramDestroy(ray_gen_pgm_orthographic_stereo) );
  }

  if ((lasterror = rtContextDestroy(context)) != RT_SUCCESS) {
    msgErr << "OptiX: An error occured while destroying the OptiX context" << sendmsg;
  }
}


void OptiXRenderer::add_material(int matindex,
                                 float ambient, float diffuse, float specular,
                                 float shininess, float reflectivity,
                                 float opacity, 
                                 float outline, float outlinewidth,
                                 int transmode) {
  int i;
  int oldmatcount = materialcache.num();
  if (oldmatcount <= matindex) {
    ort_material m;
    memset(&m, 0, sizeof(m));

    // XXX do something noticable so we see that we got a bad entry...
    m.ambient = 0.5f;
    m.diffuse = 0.7f;
    m.specular = 0.0f;
    m.shininess = 10.0f;
    m.reflectivity = 0.0f;
    m.opacity = 1.0f;
    m.transmode = 0;

    for (i=0; i<(matindex - oldmatcount + 1); i++)
      materialcache.append(m);
  }
 
  if (materialcache[matindex].isvalid) {
    return;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("Adding material[%d]\n", matindex);

    materialcache[matindex].ambient      = ambient;
    materialcache[matindex].diffuse      = diffuse; 
    materialcache[matindex].specular     = specular;
    materialcache[matindex].shininess    = shininess;
    materialcache[matindex].reflectivity = reflectivity;
    materialcache[matindex].opacity      = opacity;
    materialcache[matindex].outline      = outline;
    materialcache[matindex].outlinewidth = outlinewidth;
    materialcache[matindex].transmode    = transmode;
    materialcache[matindex].isvalid      = 1;
  }
}


void OptiXRenderer::init_materials() {
  if (verbose == RT_VERB_DEBUG) printf("OptiX: init_materials()\n");

  // pre-register all of the hit programs to be shared by all materials
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "closest_hit_radiance", &closest_hit_pgm_general ) );

#if defined(ORT_USE_TEMPLATE_SHADERS)
  // build up the list of closest hit programs from all combinations
  // of shader parameters
  int i;
  for (i=0; i<64; i++) {
    char ch_program_name[256];
    snprintf(ch_program_name, sizeof(ch_program_name),
             "closest_hit_radiance_"
             "FOG_%s_"
             "SHADOWS_%s_"
             "AO_%s_"
             "OUTLINE_%s_"
             "REFL_%s_"
             "TRANS_%s",
             onoffstr(i & 32),
             onoffstr(i & 16),
             onoffstr(i &  8),
             onoffstr(i &  4),
             onoffstr(i &  2),
             onoffstr(i &  1));

    RTERR( rtProgramCreateFromPTXFile(context, shaderpath, ch_program_name, &closest_hit_pgm_special[i] ) );
  } 
#endif

  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "any_hit_opaque", &any_hit_pgm_opaque) );
  RTERR( rtProgramCreateFromPTXFile(context, shaderpath, "any_hit_transmission", &any_hit_pgm_transmission) );

  RTERR( rtMaterialCreate(context, &material_general) );
  RTERR( rtMaterialSetClosestHitProgram(material_general, RT_RAY_TYPE_RADIANCE, closest_hit_pgm_general) );
  RTERR( rtMaterialSetAnyHitProgram(material_general, RT_RAY_TYPE_SHADOW, any_hit_pgm_transmission) );

#if defined(ORT_USE_TEMPLATE_SHADERS)
  // build up the list of materials from all combinations of shader parameters
  for (i=0; i<64; i++) {
    RTERR( rtMaterialCreate(context, &material_special[i]) );
    RTERR( rtMaterialSetClosestHitProgram(material_special[i], RT_RAY_TYPE_RADIANCE, closest_hit_pgm_special[i]) );

    // select correct any hit program depending on opacity
    if (i & 1) {
      RTERR( rtMaterialSetAnyHitProgram(material_special[i], RT_RAY_TYPE_SHADOW, any_hit_pgm_transmission) );
    } else {
      RTERR( rtMaterialSetAnyHitProgram(material_special[i], RT_RAY_TYPE_SHADOW, any_hit_pgm_opaque) );
    }
    // zero out the array of material usage counts for the scene
    material_special_counts[i] = 0;
  }
#endif

}


void OptiXRenderer::set_material(RTgeometryinstance instance, int matindex, float *uniform_color) {
//if (verbose == RT_VERB_DEBUG) printf("OptiX: setting material\n");
  RTvariable ka, kd, ks, phongexp, krefl;
  RTvariable opacity, outline, outlinewidth, transmode, uniform_col;
  RTmaterial material = material_general; 

#if defined(ORT_USE_TEMPLATE_SHADERS)
  if (getenv("VMDOPTIXFORCEGENERALSHADER") == NULL) {
    unsigned int specialized_material_index = 
      ((fog_mode != RT_FOG_NONE)                   << 5) |   // fog
      ((shadows_enabled != RT_SHADOWS_OFF)         << 4) |   // shadows
      ((ao_samples != 0)                           << 3) |   // AO
      ((materialcache[matindex].outline != 0)      << 2) |   // outline
      ((materialcache[matindex].reflectivity != 0) << 1) |   // reflection
      ((materialcache[matindex].opacity != 1)          );    // transmission

    material = material_special[specialized_material_index];

    // increment material usage counter
    material_special_counts[specialized_material_index]++;
  }
#endif

  RTERR( rtGeometryInstanceSetMaterialCount(instance, 1) );
  RTERR( rtGeometryInstanceSetMaterial(instance, 0, material) );

  if (uniform_color != NULL) {
    RTERR( rtGeometryInstanceDeclareVariable(instance, "uniform_color", &uniform_col) );
    RTERR( rtVariableSet3fv(uniform_col, uniform_color) );
  }

  RTERR( rtGeometryInstanceDeclareVariable(instance, "Ka", &ka) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "Kd", &kd) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "Ks", &ks) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "phong_exp", &phongexp) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "Krefl", &krefl) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "opacity", &opacity) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "outline", &outline) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "outlinewidth", &outlinewidth) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "transmode", &transmode) );

  RTERR( rtVariableSet1f(ka, materialcache[matindex].ambient) );
  RTERR( rtVariableSet1f(kd, materialcache[matindex].diffuse) );
  RTERR( rtVariableSet1f(ks, materialcache[matindex].specular) );
  RTERR( rtVariableSet1f(phongexp, materialcache[matindex].shininess) );
  RTERR( rtVariableSet1f(krefl, materialcache[matindex].reflectivity) );
  RTERR( rtVariableSet1f(opacity, materialcache[matindex].opacity) );
  RTERR( rtVariableSet1f(outline, materialcache[matindex].outline) );
  RTERR( rtVariableSet1f(outlinewidth, materialcache[matindex].outlinewidth) );
  RTERR( rtVariableSet1i(transmode, materialcache[matindex].transmode) );
}


void OptiXRenderer::add_directional_light(const float *dir, const float *color) {
  ort_directional_light l;
  vec_copy(l.dir, dir);
  vec_copy(l.color, color);

  directional_lights.append(l);
}


void OptiXRenderer::cylinder_array(Matrix4 *wtrans, float radius,
                                   float *uniform_color,
                                   int cylnum, float *points, int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating cylinder array: %d...\n", cylnum);
  cylinder_array_cnt += cylnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_cylinder *cyldata;

  // create and fill the OptiX cylinder array memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_cylinder));
  rtBufferSetSize1D(buf, cylnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &cyldata); // map buffer for writing by host

  if (wtrans == NULL) {
    for (i=0,ind=0; i<cylnum; i++,ind+=6) {
      // transform to eye coordinates
      vec_copy((float*) &cyldata[i].start, &points[ind]);
      cyldata[i].radius = radius;
      vec_sub((float*) &cyldata[i].axis, &points[ind+3], &points[ind]);
    }
  } else {
    for (i=0,ind=0; i<cylnum; i++,ind+=6) {
      // transform to eye coordinates
      wtrans->multpoint3d(&points[ind], (float*) &cyldata[i].start);
      cyldata[i].radius = radius;
      float ctmp[3];
      wtrans->multpoint3d(&points[ind+3], ctmp);
      vec_sub((float*) &cyldata[i].axis, ctmp, &points[ind]);
    }
  }
  rtBufferUnmap(buf); // cylinder array is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, cylnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, cylinder_array_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, cylinder_array_isct_pgm) );

  // this cyl buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "cylinder_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::cylinder_array_color(Matrix4 *wtrans, float rscale,
                                         int cylnum, float *points, 
                                         float *radii, float *colors,
                                         int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating cylinder color array: %d...\n", cylnum);
  cylinder_array_color_cnt += cylnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_cylinder_color *cyldata;

  // create and fill the OptiX cylinder array memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_cylinder_color));
  rtBufferSetSize1D(buf, cylnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &cyldata); // map buffer for writing by host

  if (wtrans == NULL) {
    // already transformed to eye coordinates
    if (radii == NULL) {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        vec_copy((float*) &cyldata[i].start, &points[ind]);
        cyldata[i].radius = rscale;
        vec_sub((float*) &cyldata[i].axis, &points[ind+3], &points[ind]);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    } else {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        vec_copy((float*) &cyldata[i].start, &points[ind]);
        cyldata[i].radius = rscale * radii[i];
        vec_sub((float*) &cyldata[i].axis, &points[ind+3], &points[ind]);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    }
  } else {
    // transform to eye coordinates
    if (radii == NULL) {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        wtrans->multpoint3d(&points[ind], (float*) &cyldata[i].start);
        cyldata[i].radius = rscale;
        float ctmp[3];
        wtrans->multpoint3d(&points[ind+3], ctmp);
        vec_sub((float*) &cyldata[i].axis, ctmp, (float*) &cyldata[i].start);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    } else {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        wtrans->multpoint3d(&points[ind], (float*) &cyldata[i].start);
        cyldata[i].radius = rscale * radii[i];
        float ctmp[3];
        wtrans->multpoint3d(&points[ind+3], ctmp);
        vec_sub((float*) &cyldata[i].axis, ctmp, (float*) &cyldata[i].start);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    }
  }
  rtBufferUnmap(buf); // cylinder array is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, cylnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, cylinder_array_color_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, cylinder_array_color_isct_pgm) );

  // this cyl buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "cylinder_color_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::ring_array_color(Matrix4 & wtrans, float rscale,
                                     int rnum, float *centers,
                                     float *norms, float *radii, 
                                     float *colors, int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating ring array color: %d...\n", rnum);
  ring_array_color_cnt += rnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_ring_color *rdata;

  // create and fill the OptiX ring array memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_ring_color));
  rtBufferSetSize1D(buf, rnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &rdata); // map buffer for writing by host

  for (i=0,ind=0; i<rnum; i++,ind+=3) {
    // transform to eye coordinates
    wtrans.multpoint3d(&centers[ind], (float*) &rdata[i].center);
    wtrans.multnorm3d(&norms[ind], (float*) &rdata[i].norm);
    vec_normalize((float*) &rdata[i].norm);
    rdata[i].inrad  = rscale * radii[i*2];
    rdata[i].outrad = rscale * radii[i*2+1];
    vec_copy((float*) &rdata[i].color, &colors[ind]);
    rdata[i].pad = 0.0f; // please valgrind  
  }
  rtBufferUnmap(buf); // ring array is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, rnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, ring_array_color_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, ring_array_color_isct_pgm) );

  // this ring buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "ring_color_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::sphere_array(Matrix4 *wtrans, float rscale,
                                 float *uniform_color,
                                 int spnum, float *centers,
                                 float *radii,
                                 int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating sphere array: %d...\n", spnum);
  sphere_array_cnt += spnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_sphere *spdata;

  // create and fill the OptiX sphere array memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_sphere));
  rtBufferSetSize1D(buf, spnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &spdata); // map buffer for writing by host

  if (wtrans == NULL) {
    if (radii == NULL) {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        vec_copy((float*) &spdata[i].center, &centers[ind]);
        spdata[i].radius = rscale; // use "rscale" as radius...
      }
    } else {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        vec_copy((float*) &spdata[i].center, &centers[ind]);
        spdata[i].radius = rscale * radii[i];
      }
    }
  } else {
    if (radii == NULL) {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        wtrans->multpoint3d(&centers[ind], (float*) &spdata[i].center);
        spdata[i].radius = rscale; // use "rscale" as radius...
      }
    } else {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        wtrans->multpoint3d(&centers[ind], (float*) &spdata[i].center);
        spdata[i].radius = rscale * radii[i];
      }
    }
  }
  rtBufferUnmap(buf); // sphere array is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, spnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, sphere_array_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, sphere_array_isct_pgm) );

  // this sphere buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "sphere_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::sphere_array_color(Matrix4 & wtrans, float rscale,
                                       int spnum, float *centers,
                                       float *radii, float *colors,
                                       int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating sphere array color: %d...\n", spnum);
  sphere_array_color_cnt += spnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_sphere_color *spdata;

  // create and fill the OptiX sphere array memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_sphere_color));
  rtBufferSetSize1D(buf, spnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &spdata); // map buffer for writing by host

  for (i=0,ind=0; i<spnum; i++,ind+=3) {
    // transform to eye coordinates
    wtrans.multpoint3d(&centers[ind], (float*) &spdata[i].center);
    spdata[i].radius = rscale * radii[i];
    vec_copy((float*) &spdata[i].color, &colors[ind]);
    spdata[i].pad = 0.0f; // please valgrind
  }
  rtBufferUnmap(buf); // sphere array is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, spnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, sphere_array_color_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, sphere_array_color_isct_pgm) );

  // this sphere buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "sphere_color_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::tricolor_list(Matrix4 & wtrans, int numtris, float *vnc,
                                  int matindex) {
//if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating tricolor list: %d...\n", numtris);
  tricolor_cnt += numtris;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numtris);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (i=0,ind=0; i<numtris; i++,ind+=27) {
    // transform to eye coordinates
    wtrans.multpoint3d(&vnc[ind     ], (float*) &trimesh[i].v0);
    wtrans.multpoint3d(&vnc[ind +  3], (float*) &trimesh[i].v1);
    wtrans.multpoint3d(&vnc[ind +  6], (float*) &trimesh[i].v2);

    wtrans.multnorm3d(&vnc[ind +  9], (float*) &trimesh[i].n0);
    wtrans.multnorm3d(&vnc[ind + 12], (float*) &trimesh[i].n1);
    wtrans.multnorm3d(&vnc[ind + 15], (float*) &trimesh[i].n2);

    vec_copy((float*) &trimesh[i].c0, &vnc[ind + 18]);
    vec_copy((float*) &trimesh[i].c1, &vnc[ind + 21]);
    vec_copy((float*) &trimesh[i].c2, &vnc[ind + 24]);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numtris) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );
//  RTERR( rtGeometryInstanceSetMaterialCount(instance, 1) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::trimesh_c4n3v3(Matrix4 & wtrans, int numverts,
                                   float *cnv, int numfacets, int * facets,
                                   int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4n3v3: %d...\n", numfacets);
  trimesh_c4u_n3b_v3f_cnt += numfacets;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (i=0,ind=0; i<numfacets; i++,ind+=3) {
    int v0 = facets[ind    ] * 10;
    int v1 = facets[ind + 1] * 10;
    int v2 = facets[ind + 2] * 10;

    // transform to eye coordinates
    wtrans.multpoint3d(cnv + v0 + 7, (float*) &trimesh[i].v0);
    wtrans.multpoint3d(cnv + v1 + 7, (float*) &trimesh[i].v1);
    wtrans.multpoint3d(cnv + v2 + 7, (float*) &trimesh[i].v2);

    wtrans.multnorm3d(cnv + v0 + 4, (float*) &trimesh[i].n0);
    wtrans.multnorm3d(cnv + v1 + 4, (float*) &trimesh[i].n1);
    wtrans.multnorm3d(cnv + v2 + 4, (float*) &trimesh[i].n2);

    vec_copy((float*) &trimesh[i].c0, cnv + v0);
    vec_copy((float*) &trimesh[i].c1, cnv + v1);
    vec_copy((float*) &trimesh[i].c2, cnv + v2);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


#if 1

// 
// This implementation translates from the most-compact host representation
// to a GPU-specific organization that balances performance vs. memory 
// storage efficiency.
//
void OptiXRenderer::trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        char *n, float *v, int numfacets, 
                                        int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3b_v3f: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_c4u_n3b_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_c4u_n3b_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    float norm[9];

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;

    // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    float3 tmpn;
    wtrans.multnorm3d(&norm[0], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n0 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[3], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n1 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[6], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n2 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);

    memcpy(&trimesh[ind].c0, &c[j  ], 4);
    memcpy(&trimesh[ind].c1, &c[j+4], 4);
    memcpy(&trimesh[ind].c2, &c[j+8], 4);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_c4u_n3b_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_c4u_n3b_v3f_isct_pgm) );

  // this trimesh buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_c4u_n3b_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}

#else

// 
// XXX this implementation unpacks into single-precision floats before
// transfer to the GPU, so it is comparatively memory inefficient.  
// It is here mainly for testing purposes. 
//
void OptiXRenderer::trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        char *n, float *v, int numfacets, 
                                        int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3b_v3f: %d...\n", numfacets);
  trimesh_c4u_n3b_v3f_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    float col[9], norm[9];

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;
    wtrans.multnorm3d(&norm[0], (float*) &trimesh[ind].n0);
    wtrans.multnorm3d(&norm[3], (float*) &trimesh[ind].n1);
    wtrans.multnorm3d(&norm[6], (float*) &trimesh[ind].n2);

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    col[0] = c[j     ] * ci2f;
    col[1] = c[j +  1] * ci2f;
    col[2] = c[j +  2] * ci2f;
    col[3] = c[j +  4] * ci2f;
    col[4] = c[j +  5] * ci2f;
    col[5] = c[j +  6] * ci2f;
    col[6] = c[j +  8] * ci2f;
    col[7] = c[j +  9] * ci2f;
    col[8] = c[j + 10] * ci2f;

    vec_copy((float*) &trimesh[ind].c0, &col[0]);
    vec_copy((float*) &trimesh[ind].c1, &col[3]);
    vec_copy((float*) &trimesh[ind].c2, &col[6]);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  // materialIndex: XXX need to do something with material properties yet...
  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}
#endif



void OptiXRenderer::trimesh_c4u_n3f_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        float *n, float *v, int numfacets, 
                                        int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3f_v3f: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    wtrans.multnorm3d(n + i    , (float*) &trimesh[ind].n0);
    wtrans.multnorm3d(n + i + 3, (float*) &trimesh[ind].n1);
    wtrans.multnorm3d(n + i + 6, (float*) &trimesh[ind].n2);

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    float col[9];
    col[0] = c[j     ] * ci2f;
    col[1] = c[j +  1] * ci2f;
    col[2] = c[j +  2] * ci2f;
    col[3] = c[j +  4] * ci2f;
    col[4] = c[j +  5] * ci2f;
    col[5] = c[j +  6] * ci2f;
    col[6] = c[j +  8] * ci2f;
    col[7] = c[j +  9] * ci2f;
    col[8] = c[j + 10] * ci2f;

    vec_copy((float*) &trimesh[ind].c0, &col[0]);
    vec_copy((float*) &trimesh[ind].c1, &col[3]);
    vec_copy((float*) &trimesh[ind].c2, &col[6]);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  // materialIndex: XXX need to do something with material properties yet...
  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::trimesh_n3b_v3f(Matrix4 & wtrans, float *uniform_color, 
                                    char *n, float *v, int numfacets, 
                                    int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_n3b_v3f: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_n3b_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_n3b_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    float norm[9];

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;

    // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    float3 tmpn;
    wtrans.multnorm3d(&norm[0], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n0 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[3], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n1 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[6], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n2 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_n3b_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_n3b_v3f_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_n3b_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::trimesh_n3f_v3f(Matrix4 & wtrans, float *uniform_color, 
                                    float *n, float *v, int numfacets, 
                                    int matindex) {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_n3f_v3f: %d...\n", numfacets);
  trimesh_n3f_v3f_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_n3f_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_n3f_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    wtrans.multnorm3d(n + i    , (float*) &trimesh[ind].n0);
    wtrans.multnorm3d(n + i + 3, (float*) &trimesh[ind].n1);
    wtrans.multnorm3d(n + i + 6, (float*) &trimesh[ind].n2);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_n3f_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_n3f_v3f_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_n3f_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


void OptiXRenderer::tristrip(Matrix4 & wtrans, int numverts, const float * cnv,
                             int numstrips, const int *vertsperstrip,
                             const int *facets, int matindex) {
  int i;
  int numfacets = 0;
  for (i=0; i<numstrips; i++) 
    numfacets += (vertsperstrip[i] - 2);  

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating tristrip: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(context, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  // loop over all of the triangle strips
  i=0; // set triangle index to 0
  for (strip=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      // transform to eye coordinates
      wtrans.multpoint3d(cnv + v0 + 7, (float*) &trimesh[i].v0);
      wtrans.multpoint3d(cnv + v1 + 7, (float*) &trimesh[i].v1);
      wtrans.multpoint3d(cnv + v2 + 7, (float*) &trimesh[i].v2);

      wtrans.multnorm3d(cnv + v0 + 4, (float*) &trimesh[i].n0);
      wtrans.multnorm3d(cnv + v1 + 4, (float*) &trimesh[i].n1);
      wtrans.multnorm3d(cnv + v2 + 4, (float*) &trimesh[i].n2);

      vec_copy((float*) &trimesh[i].c0, cnv + v0);
      vec_copy((float*) &trimesh[i].c1, cnv + v1);
      vec_copy((float*) &trimesh[i].c2, cnv + v2);

      v++; // move on to next vertex
      i++; // next triangle
    }
    v+=2; // last two vertices are already used by last triangle
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(context, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this tricolor buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(context, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  // add the newly created OptiX objects to our bookkeeping lists...
  bufferlist.append(buf);
  geomlist.append(geom);
  geominstancelist.append(instance);
}


#if !defined(VMDOPENGL)
// A hack to prevent VMD from having to be linked to libGL.so to resolve
// OptiX dependencies for OpenGL interop, e.g. when compiling on
// a supercomputer/cluster lacking OpenGL support (e.g. ORNL Titan):
//
// Linking  vmd_LINUXAMD64 ...
// /usr/lib64/libGL.so.1: undefined reference to `xcb_glx_set_client_info_arb'
// /usr/lib64/libGL.so.1: undefined reference to `xcb_glx_create_context_attribs_arb_checked'
// /usr/lib64/libGL.so.1: undefined reference to `xcb_glx_set_client_info_2arb'
// /usr/bin/ld: link errors found, deleting executable `vmd_LINUXAMD64'
// collect2: error: ld returned 1 exit status
// make: *** [vmd_LINUXAMD64] Error 1
//
extern "C" {
  typedef struct {
     unsigned int sequence;
  } xcb_void_cookie_t;
  static xcb_void_cookie_t fake_cookie = { 0 };
  xcb_void_cookie_t xcb_glx_set_client_info_arb(void) {
   return fake_cookie;
  }
  xcb_void_cookie_t xcb_glx_create_context_attribs_arb_checked(void) {
   return fake_cookie;
  }
  xcb_void_cookie_t xcb_glx_set_client_info_2arb(void) {
   return fake_cookie;
  }
}
#endif



