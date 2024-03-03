/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *               2004,2005 Wim Taymans <wim@fluendo.com>
 *
 * gstconfig.h: GST_DISABLE_* macros for build configuration
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/**
 * SECTION:gstconfig
 * @short_description: Build configuration options
 *
 * This describes the configuration options for GStreamer. When building
 * GStreamer there are a lot of parts (known internally as "subsystems" ) that
 * can be disabled for various reasons. The most common reasons are speed and
 * size, which is important because GStreamer is designed to run on embedded
 * systems.
 *
 * If a subsystem is disabled, most of this changes are done in an API
 * compatible way, so you don't need to adapt your code in most cases. It is
 * never done in an ABI compatible way though. So if you want to disable a
 * subsystem, you have to rebuild all programs depending on GStreamer, too.
 *
 * If a subsystem is disabled in GStreamer, a value is defined in
 * &lt;gst/gst.h&gt;. You can check this if you do subsystem-specific stuff.
 * <example id="example-gstconfig">
 * <title>Doing subsystem specific things</title>
 * <programlisting>
 * &hash;ifndef GST_DISABLE_GST_DEBUG
 * // do stuff specific to the debugging subsystem
 * &hash;endif // GST_DISABLE_GST_DEBUG
 * </programlisting>
 * </example>
 */

#ifndef __GST_CONFIG_H__
#define __GST_CONFIG_H__

/* trick gtk-doc into believing these symbols are defined (yes, it's ugly) */

#if 0
#define GST_DISABLE_GST_DEBUG 1
#define GST_DISABLE_PARSE 1
#define GST_DISABLE_REGISTRY 1
#define GST_DISABLE_PLUGIN 1
#endif

/***** default padding of structures *****/
#define GST_PADDING		4
#define GST_PADDING_INIT	{ NULL }

/***** padding for very extensible base classes *****/
#define GST_PADDING_LARGE	20

/***** disabling of subsystems *****/

/**
 * GST_DISABLE_GST_DEBUG:
 *
 * Configures the inclusion of the debugging subsystem
 */
#undef GST_DISABLE_GST_DEBUG

/**
 * GST_DISABLE_PARSE:
 *
 * Configures the inclusion of the gst-launch parser
 */
#undef GST_DISABLE_PARSE

/**
 * GST_DISABLE_REGISTRY:
 *
 * Configures the use of the plugin registry.
 * If one disables this, required plugins need to be loaded and registered
 * manually
 */
#undef GST_DISABLE_REGISTRY

/* FIXME: test and document these! */
/* Configures the use of external plugins */
#undef GST_DISABLE_PLUGIN

/* Whether or not the CPU supports unaligned access
 * The macros used are defined consistently by GCC, Clang, MSVC, Sun, and ICC
 *
 * References:
 * https://sourceforge.net/p/predef/wiki/Architectures/
 * https://msdn.microsoft.com/en-us/library/b0084kay.aspx
 * http://docs.oracle.com/cd/E19205-01/820-4155/c++_faq.html#Vers6
 * https://software.intel.com/en-us/node/583402
 */
#if defined(__alpha__) || defined(__arc__) || defined(__arm__) || defined(__aarch64__) || defined(__bfin) || defined(__hppa__) || defined(__nios2__) || defined(__MICROBLAZE__) || defined(__mips__) || defined(__or1k__) || defined(__sh__) || defined(__SH4__) || defined(__sparc__) || defined(__sparc) || defined(__ia64__) || defined(_M_ALPHA) || defined(_M_ARM) || defined(_M_IA64) || defined(__xtensa__) || defined(__e2k__)
#  define GST_HAVE_UNALIGNED_ACCESS 0
#elif defined(__i386__) || defined(__i386) || defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__ppc__) || defined(__ppc64__) || defined(__powerpc__) || defined(__powerpc64__) || defined(__m68k__) || defined(_M_IX86) || defined(_M_AMD64) || defined(_M_X64) || defined(__s390__) || defined(__s390x__) || defined(__zarch__)
#  define GST_HAVE_UNALIGNED_ACCESS 1
#else
#  error "Could not detect architecture; don't know whether it supports unaligned access! Please file a bug."
#endif

/**
 * GST_EXPORT:
 *
 * Export the given variable from the built shared object.
 *
 * On Windows, this exports the variable from the DLL.
 * On other platforms, this gets defined to "extern".
 */
/**
 * GST_PLUGIN_EXPORT:
 *
 * Export the plugin's definition.
 *
 * On Windows, this exports the plugin definition from the DLL.
 * On other platforms, this gets defined as a no-op.
 */
/* Only use __declspec(dllexport/import) when we have been built with MSVC or
 * the user is linking to us with MSVC. The only remaining case is when we were
 * built with MinGW and are linking with MinGW in which case we rely on the
 * linker to auto-export/import symbols. Of course all this is only used when
 * not linking statically.
 *
 * NOTE: To link to GStreamer statically on Windows, you must define
 * GST_STATIC_COMPILATION or the prototypes will cause the compiler to search
 * for the symbol inside a DLL.
 */
#if (0 || defined(_MSC_VER)) && !defined(GST_STATIC_COMPILATION)
# define GST_PLUGIN_EXPORT __declspec(dllexport)
# ifdef GST_EXPORTS
#  define GST_EXPORT __declspec(dllexport)
# else
#  define GST_EXPORT __declspec(dllimport) extern
# endif
#else
# if defined(__GNUC__) || (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590))
#  define GST_PLUGIN_EXPORT __attribute__ ((visibility ("default")))
#  define GST_EXPORT extern __attribute__ ((visibility ("default")))
# else
#  define GST_PLUGIN_EXPORT
#  define GST_EXPORT extern
# endif
#endif

#ifndef GST_API
#define GST_API GST_EXPORT
#endif

/* These macros are used to mark deprecated functions in GStreamer headers,
 * and thus have to be exposed in installed headers. But please
 * do *not* use them in other projects. Instead, use G_DEPRECATED
 * or define your own wrappers around it. */
#ifndef GST_DISABLE_DEPRECATED
#define GST_DEPRECATED GST_API
#define GST_DEPRECATED_FOR(f) GST_API
#else
#define GST_DEPRECATED G_DEPRECATED GST_API
#define GST_DEPRECATED_FOR(f) G_DEPRECATED_FOR(f) GST_API
#endif

#endif /* __GST_CONFIG_H__ */
