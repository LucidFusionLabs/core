/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

#include <QuickTime/QuickTimeComponents.h>

namespace LFL {
struct QuickTimeCameraModule : public Module {
  Thread thread;
  VideoResampler conv;
  mutex lock;
  struct Stream {
    SeqGrabComponent grabber=0;
    SGChannel channel=0;
    GWorldPtr gworld=0;
    Rect rect;
    PixMapHandle pixmap;
    ImageSequence seq=0;
    unique_ptr<RingSampler> frames;
    int next=0;
  } L, R;
  CameraState *camera;
  QuickTimeCameraModule(CameraState *C) : thread(&bind(QuickTimeCameraModule::Threadproc, this)), camera(C) {}

  int Init() {
    camera->image_format = Pixel::BGR24;
    int image_depth = Pixel::size(camera->image_format);

    /* open device */
    SGDeviceList devlist; int devind=-1, di=0, ret;
    if ((ret = EnterMovies()) != noErr) return ERRORv(-1, "EnterMovies: ", ret);
    if (!(L.grabber = OpenDefaultComponent(SeqGrabComponentType, 0))) return ERRORv(-1, "OpenDefaultComonent: ", L.grabber);
    if ((ret = SGInitialize(L.grabber)) != noErr) return ERRORv(-1, "SGInitialize: ", ret);
    if ((ret = SGSetDataRef(L.grabber, 0, 0, seqGrabDontMakeMovie)) != noErr) return ERRORv(-1, "SGSetDataRef: ", ret);
    if ((ret = SGNewChannel(L.grabber, VideoMediaType, &L.channel)) != noErr) return ERRORv(-1, "SGNewChannel: ", ret);
    if ((ret = SGGetChannelDeviceList(L.channel, 0, &devlist)) != noErr) return ERRORv(-1, "SGGetChannelDeviceList: ", ret);
    for (int i=0; i<(*devlist)->count; i++) {
      SGDeviceName devname = (*devlist)->entry[i];
      if (devname.flags) continue;
      if (FLAGS_print_camera_devices) INFO("camera device ", di, " = ", devname.name);
      if (di != FLAGS_camera_device) { di++; continue; }
      devind = i;
      break;
    }
    if (devind < 0) { FLAGS_enable_camera=0; return 0; }
    SGDeviceName devname = (*devlist)->entry[devind];
    if ((ret = SGSetChannelDevice(L.channel, devname.name)) != noErr) return ERRORv(-1, "SGSetChannelDevice: ", ret);
    if ((ret = SGDisposeDeviceList(L.grabber, devlist)) != noErr) return ERRORv(-1, "SGDisposeDeviceList: ", ret);

    /* get resolution */
    ImageDescriptionHandle imgdesc = (ImageDescriptionHandle)NewHandle(0);
    if ((ret = SGGetSrcVideoBounds(L.channel, &L.rect)) != noErr) return ERRORv(-1, "SGGetSrcVideoBounds: ", ret);
    if ((ret = QTNewGWorld(&L.gworld, k32ARGBPixelFormat, &L.rect, 0, 0, 0)) != noErr) return ERRORv(-1, "QTNewGworld: ", ret);
    if ((ret = SGSetGWorld(L.grabber, L.gworld, 0)) != noErr) return ERRORv(-1, "SGSetGWorld: ", ret);
    if ((ret = SGSetChannelBounds(L.channel, &L.rect)) != noErr) return ERRORv(-1, "SGSetChannelBounds: ", ret);
    if ((ret = SGSetChannelUsage(L.channel, seqGrabRecord)) != noErr) return ERRORv(-1, "SGSetChannelUsage: ", ret);
    if ((ret = SGStartRecord(L.grabber)) != noErr) return ERRORv(-1, "SGStartRecord: ", ret);
    if ((ret = SGGetChannelSampleDescription(L.channel, (Handle)imgdesc)) != noErr) return ERRORv(-1, "SGSetGWorld: ", ret);
    if ((ret = SGStop(L.grabber)) != noErr) return ERRORv(-1, "SGStop: ", ret);
    L.rect.right = FLAGS_camera_image_width = (*imgdesc)->width;
    L.rect.bottom = FLAGS_camera_image_height = (*imgdesc)->height;
    DisposeHandle((Handle)imgdesc);
    GWorldPtr oldgworld = L.gworld;
    if ((ret = QTNewGWorld(&L.gworld, k32ARGBPixelFormat, &L.rect, 0, 0, 0)) != noErr) return ERRORv(-1, "QTNewGworld: ", ret);
    if ((ret = SGSetGWorld(L.grabber, L.gworld, 0)) != noErr) return ERRORv(-1, "SGSetGWorld: ", ret);
    if ((ret = SGSetChannelBounds(L.channel, &L.rect)) != noErr) return ERRORv(-1, "SGSetChannelBounds: ", ret);
    DisposeGWorld(oldgworld);
    camera->image_linesize = FLAGS_camera_image_width * image_depth;
    conv.Open(FLAGS_camera_image_width, FLAGS_camera_image_height, Pixel::BGR32,
              FLAGS_camera_image_width, FLAGS_camera_image_height, camera->image_format);

    /* start */
    L.frames = make_unqiue<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, FLAGS_camera_image_width*FLAGS_camera_image_height*image_depth);
    L.pixmap = GetGWorldPixMap(L.gworld);
    LockPixels(L.pixmap);
    if ((ret = SGSetDataProc(L.grabber, NewSGDataUPP(SGDataProcL), (long)this)) != noErr) return ERRORv(-1, "SGSetDataProc: ", ret);
    if ((ret = SGStartRecord(L.grabber) )!= noErr) return ERRORv(-1, "SGSartRecord: ", ret);
    if (!thread.Start()) return -1;
    return 0;
  }

  int Free() {
    if (thread.started) thread.Wait();
    if (L.grabber) {
      SGStop(L.grabber);
      CloseComponent(L.grabber);
    }
    if (L.seq) CDSequenceEnd(L.seq);
    if (L.gworld) DisposeGWorld(L.gworld);
    return 0;
  }

  int Frame(unsigned) {
    bool new_frame = false;
    {
      ScopedMutex ML(lock);
      if (camera->frames_read > camera->last_frames_read) {

        /* frame align stream handles */
        if (L.frames) L.next = L.frames->ring.back;
        if (R.frames) R.next = R.frames->ring.back;

        new_frame = true;
      }
      camera->last_frames_read = camera->frames_read;
    }
    if (!new_frame) return 0;

    camera->image = (unsigned char*)L.frames->read(-1, L.next);
    camera->image_timestamp = L.frames->readtimestamp(-1, L.next);
    return 1;
  }

  int Threadproc() {
    while (app->run) {
      if (L.grabber) SGIdle(L.grabber);
      if (R.grabber) SGIdle(R.grabber);
      Msleep(1);
    }
  }

  OSErr SGDataProc(SGChannel chan, Ptr buf, long len, Stream *cam) {
    CodecFlags codecflags; int ret;
    if (!cam->seq) {
      ImageDescriptionHandle desc = (ImageDescriptionHandle)NewHandle(0);
      if ((ret = SGGetChannelSampleDescription(chan, (Handle)desc)) != noErr) return ERRORv(-1, "SGsFtChannelSampleDescription: ", ret);
      Rect rect; rect.top=rect.left=0; rect.right=(*desc)->width; rect.bottom=(*desc)->height;
      MatrixRecord mat; RectMatrix(&mat, &rect, &cam->rect);
      if ((ret = DecompressSequenceBegin(&cam->seq, desc, cam->gworld, 0, &cam->rect, &mat, srcCopy, 0, 0, codecNormalQuality, bestSpeedCodec)) != noErr) return ERRORv(-1, "DecompressSequenceBegin: ", ret);
      DisposeHandle((Handle)desc);
    }
    if ((ret = DecompressSequenceFrameS(cam->seq, buf, len, 0, &codecflags, 0)) != noErr) return ERRORv(-1, "DecompressSequenceFrame: ", ret);

    /* write */
    unsigned char *in = (unsigned char*)(GetPixBaseAddr(cam->pixmap)+1);
    unsigned char *out = (unsigned char*)cam->frames->Write(RingSampler::Peek | RingSampler::Stamp);
    int inlinesize = GetPixRowBytes(L.pixmap);
    conv.Resample(in, inlinesize, out, camera->image_linesize);

    /* commit */
    {
      ScopedMutex ML(lock);
      cam->frames->Write();
      camera->frames_read++;
    }
    return noErr;
  }

  static OSErr SGDataProcL(SGChannel chan, Ptr buf, long len, long *, long, TimeValue, short, long opaque) { QuickTimeCamera *cam=(QuickTimeCamera*)opaque; return cam->SGDataProc(chan, buf, len, &cam->L); }
  static OSErr SGDataProcR(SGChannel chan, Ptr buf, long len, long *, long, TimeValue, short, long opaque) { QuickTimeCamera *cam=(QuickTimeCamera*)opaque; return cam->SGDataProc(chan, buf, len, &cam->R); }
};

extern "C" void *LFAppCreateCameraModule(CameraState *state) { return new QuickTimeCameraModule(state); }

}; // namespace LFL
