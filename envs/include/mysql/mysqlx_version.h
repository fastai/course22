/*
 * Copyright (c) 2016, 2017, Oracle and/or its affiliates. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; version 2 of the
 * License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301  USA
 */

/* Version numbers for X Plugin */

#ifndef _MYSQLX_VERSION_H_
#define _MYSQLX_VERSION_H_

#define MYSQLX_PLUGIN_VERSION_MAJOR 1
#define MYSQLX_PLUGIN_VERSION_MINOR 0
#define MYSQLX_PLUGIN_VERSION_PATCH 2

#define MYSQLX_PLUGIN_NAME "mysqlx"
#define MYSQLX_STATUS_VARIABLE_PREFIX(NAME) "Mysqlx_" NAME
#define MYSQLX_SYSTEM_VARIABLE_PREFIX(NAME) "mysqlx_" NAME

#define MYSQLX_TCP_PORT             33060U
#define MYSQLX_UNIX_ADDR            "/tmp/mysqlx.sock"

#define MYSQLX_PLUGIN_VERSION ( (MYSQLX_PLUGIN_VERSION_MAJOR << 8) | MYSQLX_PLUGIN_VERSION_MINOR )
#define MYSQLX_PLUGIN_VERSION_STRING "1.0.2"

#endif // _MYSQLX_VERSION_H_
