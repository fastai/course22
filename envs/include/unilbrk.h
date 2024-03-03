/* DO NOT EDIT! GENERATED AUTOMATICALLY! */
/* Line breaking of Unicode strings.
   Copyright (C) 2001-2003, 2005-2018 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2001.

   This program is free software: you can redistribute it and/or
   modify it under the terms of either:

     * the GNU Lesser General Public License as published by the Free
       Software Foundation; either version 3 of the License, or (at your
       option) any later version.

   or

     * the GNU General Public License as published by the Free
       Software Foundation; either version 2 of the License, or (at your
       option) any later version.

   or both in parallel, as here.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef _UNILBRK_H
#define _UNILBRK_H

/* Get size_t.  */
#include <stddef.h>

#include "unitypes.h"

/* Get locale_charset() declaration.  */
#include <unistring/localcharset.h>


#ifdef __cplusplus
extern "C" {
#endif


/* These functions are locale dependent.  The encoding argument identifies
   the encoding (e.g. "ISO-8859-2" for Polish).  */


/* Line breaking.  */

enum
{
  UC_BREAK_UNDEFINED,
  UC_BREAK_PROHIBITED,
  UC_BREAK_POSSIBLE,
  UC_BREAK_MANDATORY,
  UC_BREAK_HYPHENATION
};

/* Determine the line break points in S, and store the result at p[0..n-1].
   p[i] = UC_BREAK_MANDATORY means that s[i] is a line break character.
   p[i] = UC_BREAK_POSSIBLE means that a line break may be inserted between
          s[i-1] and s[i].
   p[i] = UC_BREAK_HYPHENATION means that a hyphen and a line break may be
          inserted between s[i-1] and s[i].  But beware of language dependent
          hyphenation rules.
   p[i] = UC_BREAK_PROHIBITED means that s[i-1] and s[i] must not be separated.
 */
extern void
       u8_possible_linebreaks (const uint8_t *s, size_t n,
                               const char *encoding, char *p);
extern void
       u16_possible_linebreaks (const uint16_t *s, size_t n,
                                const char *encoding, char *p);
extern void
       u32_possible_linebreaks (const uint32_t *s, size_t n,
                                const char *encoding, char *p);
extern void
       ulc_possible_linebreaks (const char *s, size_t n,
                                const char *encoding, char *p);

/* Choose the best line breaks, assuming the uc_width function.
   The string is s[0..n-1].  The maximum number of columns per line is given
   as WIDTH.  The starting column of the string is given as START_COLUMN.
   If the algorithm shall keep room after the last piece, they can be given
   as AT_END_COLUMNS.
   o is an optional override; if o[i] != UC_BREAK_UNDEFINED, o[i] takes
   precedence over p[i] as returned by the *_possible_linebreaks function.
   The given ENCODING is used for disambiguating widths in uc_width.
   Return the column after the end of the string, and store the result at
   p[0..n-1].
 */
extern int
       u8_width_linebreaks (const uint8_t *s, size_t n, int width,
                            int start_column, int at_end_columns,
                            const char *o, const char *encoding,
                            char *p);
extern int
       u16_width_linebreaks (const uint16_t *s, size_t n, int width,
                             int start_column, int at_end_columns,
                             const char *o, const char *encoding,
                             char *p);
extern int
       u32_width_linebreaks (const uint32_t *s, size_t n, int width,
                             int start_column, int at_end_columns,
                             const char *o, const char *encoding,
                             char *p);
extern int
       ulc_width_linebreaks (const char *s, size_t n, int width,
                             int start_column, int at_end_columns,
                             const char *o, const char *encoding,
                             char *p);


#ifdef __cplusplus
}
#endif


#endif /* _UNILBRK_H */
