#include <vector>
#include <string>
using namespace std;

#include <stdlib.h>
#include <stdio.h>

#include <jni.h>
JNIEnv *jni_env;
jobject jni_activity,       jni_view,       jni_gplus;
jclass  jni_activity_class, jni_view_class, jni_gplus_class, jni_throwable_class, jni_frame_class;
jmethodID jni_view_method_swap, jni_activity_method_toggle_keyboard, jni_activity_method_write_internal_file, jni_activity_method_play_music, jni_activity_method_play_background_music, jni_gplus_method_write, jni_gplus_method_write_with_retry, jni_throwable_method_get_cause, jni_throwable_method_get_stack_trace, jni_throwable_method_tostring, jni_frame_method_tostring;

int jni_activity_width=0, jni_activity_height=0;

#include <android/log.h>
#define INFOf(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "lfjni", fmt, __VA_ARGS__)
#define CHECK(x) if (!(x)) { __android_log_print(ANDROID_LOG_ERROR, "lfjni", "%s", #x); *((int*)0)=0; }

#define ACTION_DOWN         0
#define ACTION_UP           1
#define ACTION_MOVE         2
#define ACTION_CANCEL       3
#define ACTION_OUTSIDE      4
#define ACTION_POINTER_DOWN 5
#define ACTION_POINTER_UP   6

struct Connection;
struct Service;
static Service *gplus_service = 0;

extern bool run, FLAGS_swap_axis;
extern int width, height, lfapp_opengles_version;
extern void queue_key(int button, int down);
extern void queue_mouse_click(int button, int down, int x, int y);
extern int dispatch_queued_input();
extern const char *basename(const char *text, int len=0, int *outlen=0);
extern void CmdRun(const char *name, const char *arg);
extern Connection *service_endpoint_connect(Service *svc, const char *name);
extern void service_endpoint_read(Service *svc, const char *name, const char *buf, int len);
extern void service_endpoint_close(Service *svc, const char *name);

extern int gui_gesture_swipe_up, gui_gesture_swipe_down, gui_gesture_tap[2], gui_gesture_dpad_stop[2];
extern float gui_gesture_dpad_x[2], gui_gesture_dpad_y[2], gui_gesture_dpad_dx[2], gui_gesture_dpad_dy[2];

extern "C" int main(int argc, const char **argv);

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { INFOf("%s", "OnLoad"); return JNI_VERSION_1_4; }
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_shutdown(JNIEnv* env, jclass c) {
    INFOf("%s", "shutdown");
    run = false;
}
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_main(JNIEnv *e, jclass c, jobject a) {
    CHECK(jni_env = e); jfieldID fid;
    INFOf("main: env=%p", jni_env);

    CHECK(jni_activity = jni_env->NewGlobalRef(a));
    CHECK(jni_activity_class = (jclass)jni_env->NewGlobalRef(jni_env->GetObjectClass(jni_activity)));
    CHECK(jni_activity_method_toggle_keyboard = jni_env->GetMethodID(jni_activity_class, "toggleKeyboard", "()V"));
    CHECK(jni_activity_method_play_music = jni_env->GetMethodID(jni_activity_class, "playMusic", "(Landroid/media/MediaPlayer;)V"));
    CHECK(jni_activity_method_play_background_music = jni_env->GetMethodID(jni_activity_class, "playBackgroundMusic", "(Landroid/media/MediaPlayer;)V"));
    CHECK(jni_activity_method_write_internal_file = jni_env->GetMethodID(jni_activity_class, "writeInternalFile", "(Ljava/io/FileOutputStream;[BI)V"));

    CHECK(fid = jni_env->GetFieldID(jni_activity_class, "view", "Lcom/lucidfusionlabs/lfjava/SurfaceView;"));
    CHECK(jni_view = jni_env->NewGlobalRef(jni_env->GetObjectField(jni_activity, fid)));
    CHECK(jni_view_class = (jclass)jni_env->NewGlobalRef(jni_env->GetObjectClass(jni_view)));
    CHECK(jni_view_method_swap = jni_env->GetMethodID(jni_view_class, "swapEGL", "()V"));

    CHECK(fid = jni_env->GetFieldID(jni_activity_class, "gplus", "Lcom/lucidfusionlabs/lfjava/GPlusClient;"));
    CHECK(jni_gplus = jni_env->NewGlobalRef(jni_env->GetObjectField(jni_activity, fid)));
    CHECK(jni_gplus_class = (jclass)jni_env->NewGlobalRef(jni_env->GetObjectClass(jni_gplus)));
    CHECK(jni_gplus_method_write = jni_env->GetMethodID(jni_gplus_class, "write", "(Ljava/lang/String;Ljava/nio/ByteBuffer;)V"));
    CHECK(jni_gplus_method_write_with_retry = jni_env->GetMethodID(jni_gplus_class, "writeWithRetry", "(Ljava/lang/String;Ljava/nio/ByteBuffer;)V"));

    CHECK(jni_throwable_class = jni_env->FindClass("java/lang/Throwable"));
    CHECK(jni_throwable_method_get_cause = jni_env->GetMethodID(jni_throwable_class, "getCause", "()Ljava/lang/Throwable;"));
    CHECK(jni_throwable_method_get_stack_trace = jni_env->GetMethodID(jni_throwable_class, "getStackTrace", "()[Ljava/lang/StackTraceElement;"));
    CHECK(jni_throwable_method_tostring = jni_env->GetMethodID(jni_throwable_class, "toString", "()Ljava/lang/String;"));

    CHECK(jni_frame_class = jni_env->FindClass("java/lang/StackTraceElement"));
    CHECK(jni_frame_method_tostring = jni_env->GetMethodID(jni_frame_class, "toString", "()Ljava/lang/String;"));

    int argc=1; const char *argv[2] = { "lfjni", 0 };
    int ret = main(argc, argv);
    INFOf("main: env=%p ret=%d", jni_env, ret);
}
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_resize(JNIEnv *e, jclass c, jint w, jint h) { jni_activity_width = w; jni_activity_height = h; }
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_key(JNIEnv *e, jclass c, jint down, jint keycode) { queue_key(keycode, down); }
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_touch(JNIEnv *e, jclass c, jint action, jfloat x, jfloat y, jfloat p) {
    static float lx[2]={0,0}, ly[2]={0,0};
    int dpind = FLAGS_swap_axis ? y < width/2 : x < width/2;
    if (action == ACTION_DOWN || action == ACTION_POINTER_DOWN) {
        // INFOf("%d down %f, %f", dpind, x, y);
        queue_mouse_click(1, 1, (int)x, (int)y);
        gui_gesture_tap[dpind] = 1;
        gui_gesture_dpad_x[dpind] = x;
        gui_gesture_dpad_y[dpind] = y;
        lx[dpind] = x;
        ly[dpind] = y;
    } else if (action == ACTION_UP || action == ACTION_POINTER_UP) {
        // INFOf("%d up %f, %f", dpind, x, y);
        queue_mouse_click(1, 0, (int)x, (int)y);
        gui_gesture_dpad_stop[dpind] = 1;
        gui_gesture_dpad_x[dpind] = 0;
        gui_gesture_dpad_y[dpind] = 0;
    } else if (action == ACTION_MOVE) {
        float vx = x - lx[dpind]; lx[dpind] = x;
        float vy = y - ly[dpind]; ly[dpind] = y;
        // INFOf("%d move %f, %f vel = %f, %f", dpind, x, y, vx, vy);
        if (vx > 1.5 || vx < -1.5 || vy > 1.5 || vy < -1.5) {
            gui_gesture_dpad_dx[dpind] = vx;
            gui_gesture_dpad_dy[dpind] = vy;
        }
        gui_gesture_dpad_x[dpind] = x;
        gui_gesture_dpad_y[dpind] = y;
    } else INFOf("unhandled action %d", action);
} 
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_fling(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
    int dpind = y < width/2;
    gui_gesture_dpad_dx[dpind] = vx;
    gui_gesture_dpad_dy[dpind] = vy;
    INFOf("fling(%f, %f) = %d of (%d, %d) and vel = (%f, %f)", x, y, dpind, width, height, vx, vy);
}
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_scroll(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
    gui_gesture_swipe_up = gui_gesture_swipe_down = 0;
}
extern "C" void Java_com_lucidfusionlabs_lfjava_Activity_accel(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat z) {}

extern "C" void Java_com_lucidfusionlabs_lfjava_GPlusClient_startGame(JNIEnv *e, jclass c, jboolean server, jstring pid) {
    const char *participant_id = e->GetStringUTFChars(pid, 0);
    CmdRun(server ? "gplus_server" : "gplus_client", participant_id);
    e->ReleaseStringUTFChars(pid, participant_id);
}
extern "C" void Java_com_lucidfusionlabs_lfjava_GPlusClient_read(JNIEnv *e, jclass c, jstring pid, jobject bb, jint len) {
    const char *participant_id = e->GetStringUTFChars(pid, 0);
    if (gplus_service) service_endpoint_read(gplus_service, participant_id, (const char*)e->GetDirectBufferAddress(bb), len);
    e->ReleaseStringUTFChars(pid, participant_id);
}

void android_exception_log(jthrowable &exception) {
    jobjectArray frames = (jobjectArray)jni_env->CallObjectMethod(exception, jni_throwable_method_get_stack_trace);
    jsize frames_length = jni_env->GetArrayLength(frames);
    string out;
#if 0
    if (frames > 0) {
        jstring msg = (jstring)jni_env->CallObjectMethod(exception, jni_throwable_method_tostring);
        const char *m = jni_env->GetStringUTFChars(msg, 0);
        out += m;
        jni_env->ReleaseStringUTFChars(msg, m);
        jni_env->DeleteLocalRef(msg);
    }
    for (jsize i = 0; i < frames_length; i++) { 
        jobject frame = jni_env->GetObjectArrayElement(frames, i);
        jstring msg = (jstring)jni_env->CallObjectMethod(frame, jni_frame_method_tostring);
        const char *m = jni_env->GetStringUTFChars(msg, 0);
        out += "\n    " + string(m);

        jni_env->ReleaseStringUTFChars(msg, m);
        jni_env->DeleteLocalRef(msg);
        jni_env->DeleteLocalRef(frame);
    }
    if (frames > 0) {
        jthrowable cause = (jthrowable)jni_env->CallObjectMethod(exception, jni_throwable_method_get_cause);
        if (cause) android_exception_log(cause);
    }  
#endif
    INFOf("android_exception: %s", out.c_str());
}

int android_exception() {
    jthrowable exception = jni_env->ExceptionOccurred();
    if (!exception) return 0;
    jni_env->ExceptionClear();
    android_exception_log(exception);
    return -1;
}

int android_video_init(int gles_version) {
    INFOf("%s", "android_video_init");
    width = jni_activity_width;
    height = jni_activity_height;
    const char *method_name = (gles_version == 2 ? "initEGL2" : "initEGL1");
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_view_class, method_name, "()I"));
    int ret = jni_env->CallIntMethod(jni_view, mid);
    if (ret < 0) return ret;

    lfapp_opengles_version = ret;
    return 0;
}

int android_video_swap() {
    jni_env->CallVoidMethod(jni_view, jni_view_method_swap);
    return 0;
}

int android_input(unsigned clicks, unsigned *events) {
    int ret = dispatch_queued_input();
    if (events) *events = ret;
    return 0;
}

int android_toggle_keyboard() {
    jni_env->CallVoidMethod(jni_activity, jni_activity_method_toggle_keyboard);
    return 0;
}

int android_internal_read(const char *fn, char **out, int *size) {
    jmethodID mid; jobject bytes;
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "sizeInternalFile", "(Ljava/lang/String;)I"));
    jstring jfn = jni_env->NewStringUTF(fn);
    if (!(*size = jni_env->CallIntMethod(jni_activity, mid, jfn))) return 0;
    *out = (char*)malloc(*size);
    CHECK(bytes = jni_env->NewDirectByteBuffer(*out, *size));
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "readInternalFile", "(Ljava/lang/String;[BI)I"));
    int ret = jni_env->CallIntMethod(jni_activity, mid, jfn, bytes, *size);
    jni_env->DeleteLocalRef(jfn);
    return 0;
}

void *android_internal_open_writer(const char *fn) {
    jmethodID mid; jstring jfn = jni_env->NewStringUTF(fn);
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "openInternalFileWriter", "(Ljava/lang/String;)Ljava/io/FileOutputStream;"));
    jobject handle = jni_env->CallObjectMethod(jni_activity, mid, jfn);
    jni_env->DeleteLocalRef(jfn);
    return handle;
}

int android_internal_write(void *ifw, const char *b, int l) {
    jobject handle = (jobject)ifw;
    jobject bytes = jni_env->NewDirectByteBuffer((void*)b, l);
    jni_env->CallVoidMethod(jni_activity, jni_activity_method_write_internal_file, handle, bytes, l);
    jni_env->DeleteLocalRef(bytes);
    return 0;
}

void android_internal_close_writer(void *ifw) {
    jmethodID mid; jobject handle = (jobject)ifw;
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "closeInternalFileWriter", "(Ljava/io/FileOutputStream;)V"));
    jni_env->CallVoidMethod(jni_activity, mid, handle);
}

int android_file_read(const char *fn, char **out, int *size) {
    jmethodID mid; jstring jfn = jni_env->NewStringUTF(fn);
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
    jobject assets = jni_env->CallObjectMethod(jni_activity, mid);
    jclass assets_class = jni_env->GetObjectClass(assets);

    CHECK(mid = jni_env->GetMethodID(assets_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
    jobject input = jni_env->CallObjectMethod(assets, mid, jfn);
    jni_env->DeleteLocalRef(jfn);
    jni_env->DeleteLocalRef(assets);
    jni_env->DeleteLocalRef(assets_class);
    if (!input || android_exception()) return -1;
    jclass input_class = jni_env->GetObjectClass(input);

    CHECK(mid = jni_env->GetMethodID(input_class, "available", "()I"));
    *size = jni_env->CallIntMethod(input, mid);
    jni_env->DeleteLocalRef(input_class);
    if (android_exception()) { jni_env->DeleteLocalRef(input); return -1; }
    if (!*size) { jni_env->DeleteLocalRef(input); *out=(char*)""; return 0; }

    jclass channels = jni_env->FindClass("java/nio/channels/Channels");
    CHECK(mid = jni_env->GetStaticMethodID(channels, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
    jobject readable = jni_env->CallStaticObjectMethod(channels, mid, input);
    jclass readable_class = jni_env->GetObjectClass(readable);
    jni_env->DeleteLocalRef(input);
    jni_env->DeleteLocalRef(channels);

    *out = (char*)malloc(*size);
    jobject bytes = jni_env->NewDirectByteBuffer(*out, *size);
    CHECK(mid = jni_env->GetMethodID(readable_class, "read", "(Ljava/nio/ByteBuffer;)I"));
    int ret = jni_env->CallIntMethod(readable, mid, bytes);
    jni_env->DeleteLocalRef(readable);
    jni_env->DeleteLocalRef(readable_class);
    jni_env->DeleteLocalRef(bytes);

    if (ret != *size || android_exception()) return -1;
    return 0;
}

void *android_load_music_asset(const char *fp) {
    char fn[1024];
    snprintf(fn, sizeof(fn), "%s", basename(fp,0,0));
    char *suffix = strchr(fn, '.');
    if (suffix) *suffix = 0;
    jmethodID mid; jstring jfn = jni_env->NewStringUTF(fn);
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "loadMusicAsset", "(Ljava/lang/String;)Landroid/media/MediaPlayer;"));
    jobject handle = jni_env->CallObjectMethod(jni_activity, mid, jfn);
    jni_env->DeleteLocalRef(jfn);
    return handle;
}

void android_play_music(void *h) {
    jobject handle = (jobject)h;
    jni_env->CallVoidMethod(jni_activity, jni_activity_method_play_music, handle);
}  

void android_play_background_music(void *h) {
    jobject handle = (jobject)h;
    jni_env->CallVoidMethod(jni_activity, jni_activity_method_play_background_music, handle);
}

void android_set_volume(int v) {
    jmethodID mid; jint jv = v;
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "setVolume", "(I)V"));
    return jni_env->CallVoidMethod(jni_activity, mid, jv);
}

int android_get_volume() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getVolume", "()I"));
    return jni_env->CallIntMethod(jni_activity, mid);
}

int android_get_max_volume() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "maxVolume", "()I"));
    return jni_env->CallIntMethod(jni_activity, mid);
}

int android_device_name(char *out, int size) {
    out[0] = 0;
    jmethodID mid;
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getModelName", "()Ljava/lang/String;"));
    jstring ret = (jstring)jni_env->CallObjectMethod(jni_activity, mid);
    const char *id = jni_env->GetStringUTFChars(ret, 0);
    strncpy(out, id, size-1);
    out[size-1] = 0;
    return strlen(out);
}

int android_ipv4_address() {
    jmethodID mid;
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getAddress", "()I"));
    return jni_env->CallIntMethod(jni_activity, mid);
}

int android_ipv4_broadcast_address() {
    jmethodID mid;
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getBroadcastAddress", "()I"));
    return jni_env->CallIntMethod(jni_activity, mid);
}

void android_open_browser(const char *url) {
    jmethodID mid; jstring jurl = jni_env->NewStringUTF(url);
    CHECK(mid = jni_env->GetMethodID(jni_activity_class, "openBrowser", "(Ljava/lang/String;)V"));
    jni_env->CallVoidMethod(jni_activity, mid, jurl);
    jni_env->DeleteLocalRef(jurl);
}

void android_show_ads() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "showAds", "()V"));
    jni_env->CallVoidMethod(jni_activity, mid);
}

void android_hide_ads() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "hideAds", "()V"));
    jni_env->CallVoidMethod(jni_activity, mid);
}

void android_gplus_signin() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "signIn", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
}

void android_gplus_signout() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "signOut", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
}

int android_gplus_signedin() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "signedIn", "()Z"));
    return jni_env->CallBooleanMethod(jni_gplus, mid);
}

int android_gplus_quick_game() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "quickGame", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
    return 0;
}

int android_gplus_invite() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "inviteGUI", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
    return 0;
}

int android_gplus_accept() {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "acceptGUI", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
    return 0;
}

void android_gplus_service(Service *s) { gplus_service = s; }

int android_gplus_send_unreliable(const char *participantName, const char *buf, int len) {
    jstring pn = jni_env->NewStringUTF(participantName);
    jobject bytes = jni_env->NewDirectByteBuffer((void*)buf, len);
    jni_env->CallVoidMethod(jni_gplus, jni_gplus_method_write, pn, bytes);
    jni_env->DeleteLocalRef(bytes);
    jni_env->DeleteLocalRef(pn);
    return 0;
}

int android_gplus_send_reliable(const char *participantName, const char *buf, int len) {
    jstring pn = jni_env->NewStringUTF(participantName);
    jobject bytes = jni_env->NewDirectByteBuffer((void*)buf, len);
    jni_env->CallVoidMethod(jni_gplus, jni_gplus_method_write_with_retry, pn, bytes);
    jni_env->DeleteLocalRef(bytes);
    jni_env->DeleteLocalRef(pn);
    return 0;
}
