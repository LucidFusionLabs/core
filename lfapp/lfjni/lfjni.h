/*
 * $Id: lfapp.h 770 2013-09-25 00:27:33Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __LFL_LFJNI_LFJNI_H__
#define __LFL_LFJNI_LFJNI_H__

struct Service;
struct Connection;

int android_video_init(int gles_version);
int android_video_swap();
int android_input(unsigned clicks, unsigned *events);
int android_toggle_keyboard();
int android_file_read(const char *filename, char **mallocOut, int *sizeOut);
int android_internal_read(const char *fn, char **mallocOut, int *sizeOut);
void *android_internal_open_writer(const char *fn);
int android_internal_write(void *ifw, const char *b, int l);
void android_internal_close_writer(void *ifw);
void *android_load_music_asset(const char *filename);
void android_play_music(void *handle);
void android_play_background_music(void *handle);
void android_set_volume(int v);
int android_get_volume();
int android_get_max_volume();
int android_device_name(char *out, int len);
int android_ipv4_address();
int android_ipv4_broadcast_address();
void android_open_browser(const char *url);
void android_show_ads();
void android_hide_ads();
void android_gplus_signin();
void android_gplus_signout();
int android_gplus_signedin();
int android_gplus_invite();
int android_gplus_accept();
int android_gplus_quick_game();
void android_gplus_service(Service *s);
int android_gplus_send_unreliable(const char *participantName, const char *buf, int len);
int android_gplus_send_reliable(const char *participantName, const char *buf, int len);

#endif
