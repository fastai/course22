/* Character set conversion handler type.
   Copyright (C) 2001-2007, 2009-2018 Free Software Foundation, Inc.
   Written by Bruno Haible.

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
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef _ICONVEH_H
#define _ICONVEH_H


#ifdef __cplusplus
extern "C" {
#endif


/* Handling of unconvertible characters.  */
enum iconv_ilseq_handler
{
  iconveh_error,                /* return and set errno = EILSEQ */
  iconveh_question_mark,        /* use one '?' per unconvertible character */
  iconveh_escape_sequence       /* use escape sequence \uxxxx or \Uxxxxxxxx */
};


#ifdef __cplusplus
}
#endif


#endif /* _ICONVEH_H */
