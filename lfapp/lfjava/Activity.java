package com.lucidfusionlabs.lfjava;

import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import android.os.*;
import android.view.*;
import android.widget.FrameLayout;
import android.widget.FrameLayout.LayoutParams;
import android.media.*;
import android.content.*;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.net.Uri;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.*;

import com.google.example.games.basegameutils.*;
import com.google.android.gms.games.Games;
import com.google.android.gms.games.GamesStatusCodes;
import com.google.android.gms.games.multiplayer.*;
import com.google.android.gms.games.multiplayer.realtime.*;
import com.google.android.gms.ads.*;

class GPlusClient implements GameHelper.GameHelperListener, OnInvitationReceivedListener, RoomStatusUpdateListener, RoomUpdateListener, RealTimeMessageReceivedListener {
    public static native void startGame(boolean server, String participant_id);
    public static native void read(String participant_id, java.nio.ByteBuffer buf, int len);
 
    public static final int RC_SELECT_PLAYERS = 10000;
    public static final int RC_INVITATION_INBOX = 10001;
    public static final int RC_WAITING_ROOM = 10002;

    public Activity activity;
    public GameHelper google;
    public boolean google_signed_in, server_role, server_started, auto_match;
    public String room_id, server_pid;
    public List<String> room_pids;
    public ByteBuffer read_buffer;

    public GPlusClient(Activity act) {
        activity = act;
        google = new GameHelper(activity, GameHelper.CLIENT_GAMES);
        google.setup(this);
        read_buffer = ByteBuffer.allocateDirect(4096);
    }
    
    public void onStart(Activity act) { google.onStart(act); }
    public void onStop() { leaveRoom(); google.onStop(); }
    
    @Override
    public void onSignInFailed() { google_signed_in = false; }
    
    @Override
    public void onSignInSucceeded() {
        google_signed_in = true;
        Games.Invitations.registerInvitationListener(google.getApiClient(), this);
        if (google.getInvitationId() != null) {
            acceptInvitation(google.getInvitationId());
        }
    }

    public void onActivityResult(int request, int response, Intent data) {
        if (google != null) { google.onActivityResult(request, response, data); }
        
        if (request == RC_INVITATION_INBOX) {
            if (response != Activity.RESULT_OK) { return; }
            Invitation invitation = data.getExtras().getParcelable(Multiplayer.EXTRA_INVITATION);
            acceptInvitation(invitation.getInvitationId());
        }
        else if (request == RC_SELECT_PLAYERS) {
            if (response != Activity.RESULT_OK) { return; }
            RoomConfig.Builder roomConfigBuilder = makeBasicRoomConfigBuilder();

            final ArrayList<String> invitees = data.getStringArrayListExtra(Games.EXTRA_PLAYER_IDS);
            roomConfigBuilder.addPlayersToInvite(invitees);

            int minAutoMatchPlayers = data.getIntExtra(Multiplayer.EXTRA_MIN_AUTOMATCH_PLAYERS, 0);
            int maxAutoMatchPlayers = data.getIntExtra(Multiplayer.EXTRA_MAX_AUTOMATCH_PLAYERS, 0);
            if (minAutoMatchPlayers > 0) {
                roomConfigBuilder.setAutoMatchCriteria(RoomConfig.createAutoMatchCriteria(minAutoMatchPlayers, maxAutoMatchPlayers, 0));
            }

            createRoom(roomConfigBuilder.build());
        }
        else if (request == RC_WAITING_ROOM) {
            if (response != Activity.RESULT_OK) { leaveRoom(); return; }
            Log.i("lfjava", "startGame(" + server_role + ", " + server_pid + ")");
            startGame(server_role, server_pid);
        }
    }

    @Override
    public void onInvitationReceived(Invitation invitation) {
        Log.i("lfjava", "GplusClient::onInivitationReceived(" + invitation.getInvitationId() + ")");
        // show in-game popup to let user know of pending invitation
        // store invitation for use when player accepts this invitation
        // mIncomingInvitationId = invitation.getInvitationId();
    }
    
    @Override
    public void onInvitationRemoved(String arg0) {
        Log.i("lfjava", "GPlusClient::onInvitationRemoved " + arg0);
    }
    
    @Override
    public void onPeerInvitedToRoom(Room arg0, List<String> arg1) {}
    
    @Override
    public void onPeerDeclined(Room room, List<String> peers) {
        Log.i("lfjava", "GPlusClient::onPeerDeclined " + peers);
    }
    
    @Override
    public void onPeerJoined(Room arg0, List<String> peers) {
        Log.i("lfjava", "GPlusClient::onPeerJoined " + peers);
    }
    
    @Override
    public void onPeerLeft(Room room, List<String> peers) {
        Log.i("lfjava", "GPlusClient::onPeerLeft " + peers);
    }
    
    @Override
    public void onPeersConnected(Room room, List<String> peers) {
        Log.i("lfjava", "GPlusClient::onPeersConnected " + peers);
    }
    
    @Override
    public void onPeersDisconnected(Room room, List<String> peers) {
        Log.i("lfjava", "GPlusClient::onPeersDisconnected " + peers);
    }
    
    @Override
    public void onConnectedToRoom(Room room) {
        Log.i("lfjava", "GPlusClient::onConnectedToRoom " + room.getRoomId());
    }

    @Override
    public void onDisconnectedFromRoom(Room room) {
        Log.i("lfjava", "GPlusClient::onDisconnectedFromRoom " + room.getRoomId());
        if (room_id != null) leaveRoom();
    }
 
    @Override
    public void onP2PConnected(String participantId) {}
 
    @Override
    public void onP2PDisconnected(String participantId) {}
    
    @Override
    public void onRoomAutoMatching(Room room) {}

    @Override
    public void onRoomConnecting(Room room) {}

    @Override
    public void onRoomConnected(int statusCode, Room room) {
        Log.i("lfjava", "onRoomConnected(" + statusCode + ", " + room.getRoomId() + ") All participants connected");
        String my_pid = room.getParticipantId(Games.Players.getCurrentPlayerId(google.getApiClient()));
        room_pids = room.getParticipantIds();
        Collections.sort(room_pids);
        server_pid = room_pids.get(0);
        server_role = server_pid.compareTo(my_pid) == 0;
        Log.i("lfjava", "server_pid=" + server_pid + ", my_pid=" + my_pid);
    }
    
    @Override
    public void onLeftRoom(int statusCode, String roomId) {
        Log.i("lfjava", "GPlusClient::onLeftRoom(" + statusCode + ", " + roomId + ")");
    }
    
    @Override
    public void onJoinedRoom(int statusCode, Room room) {
        if (statusCode != GamesStatusCodes.STATUS_OK) return;
        Log.i("lfjava", "GPlusClient.onJoinedRoom(" + room.getRoomId() + ")");
        if (room_id != null) leaveRoom();
        room_id = room.getRoomId();
        
        activity.waiting_activity_result = true;
        activity.startActivityForResult(Games.RealTimeMultiplayer.getWaitingRoomIntent(google.getApiClient(), room, Integer.MAX_VALUE), RC_WAITING_ROOM);
    }

    @Override
    public void onRoomCreated(int statusCode, Room room) {
        if (statusCode != GamesStatusCodes.STATUS_OK) { return; }
        Log.i("lfjava", "GPlusClient.onRoomCreated(" + room.getRoomId() + ")");
        if (room_id != null) leaveRoom();
        room_id = room.getRoomId();

        activity.waiting_activity_result = true;
        activity.startActivityForResult(Games.RealTimeMultiplayer.getWaitingRoomIntent(google.getApiClient(), room, Integer.MAX_VALUE), RC_WAITING_ROOM);
    }
    
    @Override
    public void onRealTimeMessageReceived(RealTimeMessage message) {
        int len = message.getMessageData().length;
        // Log.i("lfjava", "onRealTimeMessageReceived " + message.getSenderParticipantId() + " " + len);

        if (len > read_buffer.capacity()) {
            Log.e("lfjava", len + " > " + read_buffer.capacity());
            return;
        }

        read_buffer.clear();
        read_buffer.put(message.getMessageData(), 0, len);
        read_buffer.flip();
        read(message.getSenderParticipantId(), read_buffer, len);
    }
    
    public RoomConfig.Builder makeBasicRoomConfigBuilder() {
        RoomConfig.Builder roomConfigBuilder = RoomConfig.builder(this);
        roomConfigBuilder.setMessageReceivedListener(this);
        roomConfigBuilder.setRoomStatusUpdateListener(this);
        return roomConfigBuilder;
    }
    
    public void acceptInvitation(String id) {
        RoomConfig.Builder roomConfigBuilder = makeBasicRoomConfigBuilder();
        roomConfigBuilder.setInvitationIdToAccept(id);
        joinRoom(roomConfigBuilder.build());
    }
    
    public void joinRoom(RoomConfig config) {
        if (room_id != null) leaveRoom();
        Games.RealTimeMultiplayer.join(google.getApiClient(), config);
        server_role = false;
    }
    
    public void createRoom(RoomConfig config) {
        if (room_id != null) leaveRoom();
        Games.RealTimeMultiplayer.create(google.getApiClient(), config); 
        server_role = true;
    }
    
    public void leaveRoom() {
        if (room_id == null) return;
        Log.i("lfjava", "GPlusClient.leaveRoom(" + room_id + ")");
        Games.RealTimeMultiplayer.leave(google.getApiClient(), this, room_id);
        room_id = null;
        server_role = false;
        server_started = false;
        auto_match = false;
        server_pid = null;
        room_pids = null;
    }
    
    public boolean signedIn() { return google_signed_in; }
    public void signIn() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                activity.waiting_activity_result = true;
                google.beginUserInitiatedSignIn();
            }
        });
    }
    public void signOut() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                google.signOut();
                google_signed_in = false;
            }
        });
    }
       
    public void quickGame() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                RoomConfig.Builder roomConfigBuilder = makeBasicRoomConfigBuilder();
                roomConfigBuilder.setAutoMatchCriteria(RoomConfig.createAutoMatchCriteria(1, 1, 0));
                createRoom(roomConfigBuilder.build());
            }
        });
    }
    public void inviteGUI() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                Intent intent = Games.RealTimeMultiplayer.getSelectOpponentsIntent(google.getApiClient(), 1, 3);
                activity.waiting_activity_result = true;
                activity.startActivityForResult(intent, RC_SELECT_PLAYERS);
            }
        });
    }
    public void acceptGUI() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                Intent intent = Games.Invitations.getInvitationInboxIntent(google.getApiClient());
                activity.waiting_activity_result = true;
                activity.startActivityForResult(intent, RC_INVITATION_INBOX);
            }
        });
    }
    
    class Write implements Runnable {
        final String recipientParticipantId;
        final ByteBuffer messageData;
        boolean reliable;

        Write(String pid, ByteBuffer buf, boolean retry) {
            recipientParticipantId = new String(pid);
            messageData = ByteBuffer.allocate(buf.capacity());
            messageData.put(buf);
            messageData.flip();
            reliable = retry;
        }
        public void run() {
            if (reliable) RunReliable();
            else          RunUnreliable();
        }
        public void RunReliable() {
            if (room_id == null) return;
            // Log.i("lfjava", "sendReliable " + room_id + " " + recipientParticipantId);
            if (Games.RealTimeMultiplayer.sendReliableMessage(google.getApiClient(), null, messageData.array(), room_id, recipientParticipantId) < 0)
                Log.e("lfjava", "GPlusClient.sendReliable(" + room_id + ", " + recipientParticipantId + ")");
        }
        public void RunUnreliable() {
            if (room_id == null) return;
            // Log.i("lfjava", "sendUnreliable " + room_id + " " + recipientParticipantId);
            if (Games.RealTimeMultiplayer.sendUnreliableMessage(google.getApiClient(), messageData.array(), room_id, recipientParticipantId) < 0)
                Log.e("lfjava", "GPlusClient.sendUnreliable(" + room_id + ", " + recipientParticipantId + ")");
        }
    }
    
    public void write         (String recipientParticipantId, ByteBuffer messageData) { activity.runOnUiThread(new Write(recipientParticipantId, messageData, false)); }
    public void writeWithRetry(String recipientParticipantId, ByteBuffer messageData) { activity.runOnUiThread(new Write(recipientParticipantId, messageData, true )); }
}

public class Activity extends android.app.Activity {
    static { System.loadLibrary("lfjni"); }
    
    public static native void main(Object activity);
    public static native void shutdown();
    public static native void resize(int x, int y);
    public static native void key(int upordonw, int keycode);
    public static native void touch(int action, float x, float y, float p);
    public static native void fling(float x, float y, float vx, float vy);
    public static native void scroll(float x, float y, float sx, float sy);
    public static native void accel(float x, float y, float z);

    public static Activity instance;
    public FrameLayout frameLayout;
    public SurfaceView view;
    public Thread thread;
    public AdView adView;
    public AdRequest adRequest;
    public GPlusClient gplus;
    public AudioManager audio;
    public boolean waiting_activity_result;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("lfjava", "Activity.onCreate()");

        instance = this;
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        super.onCreate(savedInstanceState);
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                                  WindowManager.LayoutParams.FLAG_FULLSCREEN);
        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);        

        // final ActivityManager activityManager = (ActivityManager)getSystemService(Context.ACTIVITY_SERVICE);
        // final ConfigurationInfo configurationInfo = activityManager.getDeviceConfigurationInfo();
        // final boolean supportsEs2 = configurationInfo.reqGlEsVersion >= 0x20000;
        // if (supportsEs2) {}        
        
        frameLayout = new FrameLayout(this);
        
        view = new SurfaceView(this, getApplication());
        view.getHolder().setType(SurfaceHolder.SURFACE_TYPE_GPU);
 
        adView = new AdView(this);
        adView.setAdSize(AdSize.BANNER);
        adView.setAdUnitId("ca-app-pub-6299262366078490/8570530772");

        frameLayout.addView(view, new LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT));
        frameLayout.addView(adView, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT, Gravity.TOP + Gravity.CENTER_HORIZONTAL));
        setContentView(frameLayout, new LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT));

        gplus = new GPlusClient(this);
        audio = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
    }
    
    @Override
    protected void onDestroy() {
        Log.i("lfjava", "Activity.onDestroy()");
        if (adView != null) { adView.destroy(); adView=null; } 
        super.onDestroy();
        close();
        exit();
    }

    @Override
    protected void onPause() {
        Log.i("lfjava", "Activity.onPause()");
        super.onPause();
        if (!waiting_activity_result) { close(); }
    }

    @Override
    protected void onStart() {
        Log.i("lfjava", "Activity.onStart()");
        super.onStart();
        if (gplus != null) gplus.onStart(this);
    }

    @Override
    protected void onStop() {
        Log.i("lfjava", "Activity.onStop()");
        super.onStop();
        if (gplus != null) gplus.onStop();
    }
    
    @Override
    protected void onActivityResult(int request, int response, Intent data) {
        Log.i("lfjava", "Activity.onActivityResult(" + request + ", " + response + ")");
        waiting_activity_result = false;
        super.onActivityResult(request, response, data);
        if (gplus != null) gplus.onActivityResult(request, response, data);
    }
    
    void exit() {
        Log.i("lfjava", "Activity.exit()");
        android.os.Process.killProcess(android.os.Process.myPid());
    }

    void open(int format, int width, int height) {
        Log.i("lfjava", "Activity.open()");
        resize(width, height);

        if (thread != null) return;
        thread = new Thread(new Runnable() { public void run() {
            Activity.main(Activity.instance);
            Activity.instance.finish();
        } }, "JNIMainThread");
        thread.start();
    }

    void close() {
        if (thread == null) return;
        Log.i("lfjava", "Activity.close() enter");
        Thread t = thread;
        thread = null;        
        shutdown();
        try { t.join(); }
        catch(Exception e) { Log.e("lfjava", e.toString()); }
        Log.i("lfjava", "Activity.close() exit");
    }

    public int readInternalFile(String filename, byte[] buf, int size) {
        try {
            java.io.FileInputStream inputStream = openFileInput(filename);
            inputStream.read(buf, 0, size);
            inputStream.close();
            return size;
        } catch (final Exception e) { Log.e("lfjava", e.toString()); return 0; }
    }
    public int sizeInternalFile(String filename) {
        try {
            java.io.File file = new java.io.File(getFilesDir(), filename);
            return file.exists() ? (int)file.length() : 0;
        } catch (final Exception e) { Log.e("lfjava", e.toString()); return 0; }
    }
    public java.io.FileOutputStream openInternalFileWriter(String filename) {
        try { return openFileOutput(filename, Context.MODE_PRIVATE); } 
        catch (final Exception e) { Log.e("lfjava", e.toString()); return null; }
    }
    public void writeInternalFile(java.io.FileOutputStream outputStream, byte[] buf, int size) {
        try { outputStream.write(buf, 0, size); }
        catch (final Exception e) { Log.e("lfjava", e.toString()); }
    }
    public void closeInternalFileWriter(java.io.FileOutputStream outputStream) {
        try { outputStream.close(); }
        catch (final Exception e) { Log.e("lfjava", e.toString()); }
    }

    public void toggleKeyboard() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.toggleSoftInput(android.view.inputmethod.InputMethodManager.SHOW_FORCED, 0);
    }

    public MediaPlayer loadMusicAsset(String filename) {
        android.content.res.Resources res = getResources();
        String package_name = getPackageName();
        int soundId = res.getIdentifier(filename, "raw", getPackageName());
        Log.i("lfjava", "loadMusicAsset " + package_name + " " + filename + " " + soundId);
        return MediaPlayer.create(this, soundId);
    }

    public void playMusic(MediaPlayer mp) {
        mp.start();
    }

    public void playBackgroundMusic(MediaPlayer mp) {
        mp.setLooping(true);
        mp.start();
    }

    public int maxVolume() { return audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC); }
    public int getVolume() { return audio.getStreamVolume(AudioManager.STREAM_MUSIC); }
    public void setVolume(int v) { audio.setStreamVolume(AudioManager.STREAM_MUSIC, v, 0); }
    
    public String getModelName() {
        return android.os.Build.MANUFACTURER + " " + android.os.Build.MODEL;
    }
    
    public int getAddress() {
        int ret = 0;
        try {
            for (java.util.Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (java.util.Enumeration<InetAddress> IpAddresses = intf.getInetAddresses(); IpAddresses.hasMoreElements();) {
                    InetAddress inetAddress = IpAddresses.nextElement();
                    if (inetAddress.isLoopbackAddress() || inetAddress instanceof Inet6Address) continue;
                    ret = ByteBuffer.wrap(inetAddress.getAddress()).getInt();
                }
            }  
        } catch(Exception e) {}
        return ret;
    }

    public int getBroadcastAddress() {
        int ret = 0;
        try {
            for (java.util.Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (java.util.Enumeration<InetAddress> IpAddresses = intf.getInetAddresses(); IpAddresses.hasMoreElements();) {
                    InetAddress inetAddress = IpAddresses.nextElement(), bcastAddress = null;
                    if (inetAddress.isLoopbackAddress() || inetAddress instanceof Inet6Address) continue;
                    // Log.i("lfjava", "ip address: " + inetAddress);
                    NetworkInterface ipv4_intf = NetworkInterface.getByInetAddress(inetAddress);
                    for (InterfaceAddress ifaceAddress : ipv4_intf.getInterfaceAddresses()) bcastAddress = ifaceAddress.getBroadcast();
                    // Log.i("lfjava", "broadcast aaddress: " + bcastAddress);
                    ret = ByteBuffer.wrap(bcastAddress.getAddress()).getInt();   
                }
            }  
        } catch(Exception e) {}
        return ret;
    }

    public void openBrowser(String url) {
        final Intent intent = new Intent(Intent.ACTION_VIEW).setData(Uri.parse(url));
        startActivity(intent);
    }

    public void hideAds() {
        runOnUiThread(new Runnable() {
            public void run() {
                adView.setVisibility(AdView.INVISIBLE);
            }
        });
    }
    public void showAds() {
        runOnUiThread(new Runnable() {
            public void run() {
                if (adRequest == null) { 
                    Log.i("lfjava", "creating adRequest");
                    adRequest = new AdRequest.Builder()
                        .addTestDevice("0DBA42FB5610516D099B960C0424B343")
                        .addTestDevice("52615418751B72981EF42A9681544EDB")
                        .build();
                    adView.loadAd(adRequest);
                } else {
                    adView.setVisibility(AdView.VISIBLE);
                    adView.bringToFront();
                }
            }
        });
    }
}

class MyGestureListener extends android.view.GestureDetector.SimpleOnGestureListener {
    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        Activity.fling(e1.getX(), e1.getY(), velocityX, velocityY);
        return false;
    }
}

class SurfaceView extends android.view.SurfaceView implements SurfaceHolder.Callback, View.OnKeyListener, View.OnTouchListener, SensorEventListener {

    public EGLContext egl_context;
    public EGLSurface egl_surface;
    public EGLDisplay egl_display;
    public GestureDetector gesture;
    public SensorManager sensor;

    public SurfaceView(Activity activity, Context context) {
        super(context);
        // gesture = new GestureDetector(new MyGestureListener());
        sensor = (SensorManager)context.getSystemService("sensor");  
        setFocusable(true);
        setFocusableInTouchMode(true);
        setOnKeyListener(this); 
        setOnTouchListener(this);   
        getHolder().addCallback(this); 
        requestFocus();
    }

    public void surfaceCreated(SurfaceHolder holder) {
       sensor.registerListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME, null);
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("lfjava", "SurfaceView.surfaceChanged(" + width + ", " + height + ")");
        Activity.instance.open(format, width, height);
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("lfjava", "surfaceDestroyed()");
        Activity.instance.close(); 
        sensor.unregisterListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
    }

    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) Activity.accel(event.values[0], event.values[1], event.values[2]);
    }

    public boolean onKey(View v, int keyCode, KeyEvent event) {
        int keyChar = event.getUnicodeChar();
        if      (event.getAction() == KeyEvent.ACTION_UP)   { Activity.key(0, keyChar); return true; }
        else if (event.getAction() == KeyEvent.ACTION_DOWN) { Activity.key(1, keyChar); return true; }
        else return false;
    }

    public boolean onTouch(View v, MotionEvent event) {
        // gesture.onTouchEvent(event);
        final int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_MOVE) {
            for (int i = 0; i < event.getPointerCount(); i++) {
                Activity.touch(action, event.getX(i), event.getY(i), event.getPressure(i));
            }
        } else {
            int action_index = (action == MotionEvent.ACTION_POINTER_DOWN || action == MotionEvent.ACTION_POINTER_UP) ? event.getActionIndex() : 0;
            Activity.touch(action, event.getX(action_index), event.getY(action_index), event.getPressure(action_index));
        }
        return true; 
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    public boolean initEGL(int requestedVersion) {
        try {
            final int EGL_OPENGL_ES2_BIT = 4, EGL_CONTEXT_CLIENT_VERSION = 0x3098;
            final int[] attrib_list1 = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_NONE };
            final int[] attrib_list2 = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL10.EGL_NONE };
            final int[] context_attrib2 = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL10.EGL_NONE };
            int[] attrib_list = requestedVersion >= 2 ? attrib_list2 : attrib_list1;
            int[] context_attrib = requestedVersion >= 2 ? context_attrib2 : null;
            int[] version = new int[2], num_config = new int[1];

        	EGLConfig[] configs = new EGLConfig[1];
        	EGL10 egl = (EGL10)EGLContext.getEGL();
            egl_display = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
            egl.eglInitialize(egl_display, version);

            if (!egl.eglChooseConfig(egl_display, attrib_list, configs, 1, num_config) || num_config[0] == 0) throw new Exception("eglChooseConfig");
            if ((egl_context = egl.eglCreateContext(egl_display, configs[0], EGL10.EGL_NO_CONTEXT, context_attrib)) == EGL10.EGL_NO_CONTEXT) throw new Exception("eglCreateContext");
            if ((egl_surface = egl.eglCreateWindowSurface(egl_display, configs[0], this, null)) == EGL10.EGL_NO_SURFACE) throw new Exception("eglCreateWindowSurface");
            if (!egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) throw new Exception("eglMakeCurrent");
        }
        catch(Exception e) {
            Log.e("lfjava", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfjava", ste.toString());
            egl_context = null; egl_surface = null; egl_display = null;
            return false;
        }
        return true;
    }
    
    public int initEGL1() {
        if (initEGL(1)) return 1;
        return -1;
    }
    public int initEGL2() {
        if (initEGL(2)) return 2;
        if (initEGL(1)) return 1;
        return -1;
    }

    public void swapEGL() {
        try {
            EGL10 egl = (EGL10)EGLContext.getEGL();
            // egl.eglWaitNative(EGL10.EGL_NATIVE_RENDERABLE, null);
            egl.eglWaitNative(EGL10.EGL_CORE_NATIVE_ENGINE, null);
            egl.eglWaitGL();
            egl.eglSwapBuffers(egl_display, egl_surface);
        }
        catch(Exception e) {
            Log.e("lfjava", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfjava", ste.toString());
        }
    }    
}
