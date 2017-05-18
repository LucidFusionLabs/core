package com.lucidfusionlabs.app;

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

import com.google.android.gms.games.Games;
import com.google.android.gms.games.GamesStatusCodes;
import com.google.android.gms.games.multiplayer.*;
import com.google.android.gms.games.multiplayer.realtime.*;

import com.google.example.games.basegameutils.*;
import com.lucidfusionlabs.core.LifecycleActivity;

class GPlusClient extends com.lucidfusionlabs.core.ActivityLifecycleListener implements
    GameHelper.GameHelperListener, OnInvitationReceivedListener, RoomStatusUpdateListener,
    RoomUpdateListener, RealTimeMessageReceivedListener {

    public static native void startGame(boolean server, String participant_id);
    public static native void read(String participant_id, java.nio.ByteBuffer buf, int len);
 
    public static final int RC_SELECT_PLAYERS = 10000;
    public static final int RC_INVITATION_INBOX = 10001;
    public static final int RC_WAITING_ROOM = 10002;

    public LifecycleActivity activity;
    public GameHelper google;
    public boolean google_signed_in, server_role, server_started, auto_match;
    public String room_id, server_pid;
    public List<String> room_pids;
    public ByteBuffer read_buffer;

    public GPlusClient(LifecycleActivity act) {
        activity = act;
        activity.registerLifecycleListener(this);
        google = new GameHelper(activity, GameHelper.CLIENT_GAMES);
        google.setup(this);
        read_buffer = ByteBuffer.allocateDirect(4096);
    }
    
    @Overrride
    public void onActivityStarted(Activity act) { google.onStart(act); }

    @Overrride
    public void onActivityStopped(Activity act) { leaveRoom(); google.onStop(); }

    @Override 
    public void onActivityResult(Activity act, int request, int response, Intent data) {
        if (google != null) { google.onActivityResult(request, response, data); }
        
        if (request == RC_INVITATION_INBOX) {
            if (response != MainActivity.RESULT_OK) { return; }
            Invitation invitation = data.getExtras().getParcelable(Multiplayer.EXTRA_INVITATION);
            acceptInvitation(invitation.getInvitationId());
        }
        else if (request == RC_SELECT_PLAYERS) {
            if (response != MainActivity.RESULT_OK) { return; }
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
            if (response != MainActivity.RESULT_OK) { leaveRoom(); return; }
            Log.i("lfl", "startGame(" + server_role + ", " + server_pid + ")");
            startGame(server_role, server_pid);
        }
    }
    
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

    @Override
    public void onInvitationReceived(Invitation invitation) {
        Log.i("lfl", "GplusClient::onInivitationReceived(" + invitation.getInvitationId() + ")");
        // show in-game popup to let user know of pending invitation
        // store invitation for use when player accepts this invitation
        // mIncomingInvitationId = invitation.getInvitationId();
    }
    
    @Override
    public void onInvitationRemoved(String arg0) {
        Log.i("lfl", "GPlusClient::onInvitationRemoved " + arg0);
    }
    
    @Override
    public void onPeerInvitedToRoom(Room arg0, List<String> arg1) {}
    
    @Override
    public void onPeerDeclined(Room room, List<String> peers) {
        Log.i("lfl", "GPlusClient::onPeerDeclined " + peers);
    }
    
    @Override
    public void onPeerJoined(Room arg0, List<String> peers) {
        Log.i("lfl", "GPlusClient::onPeerJoined " + peers);
    }
    
    @Override
    public void onPeerLeft(Room room, List<String> peers) {
        Log.i("lfl", "GPlusClient::onPeerLeft " + peers);
    }
    
    @Override
    public void onPeersConnected(Room room, List<String> peers) {
        Log.i("lfl", "GPlusClient::onPeersConnected " + peers);
    }
    
    @Override
    public void onPeersDisconnected(Room room, List<String> peers) {
        Log.i("lfl", "GPlusClient::onPeersDisconnected " + peers);
    }
    
    @Override
    public void onConnectedToRoom(Room room) {
        Log.i("lfl", "GPlusClient::onConnectedToRoom " + room.getRoomId());
    }

    @Override
    public void onDisconnectedFromRoom(Room room) {
        Log.i("lfl", "GPlusClient::onDisconnectedFromRoom " + room.getRoomId());
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
        Log.i("lfl", "onRoomConnected(" + statusCode + ", " + room.getRoomId() + ") All participants connected");
        String my_pid = room.getParticipantId(Games.Players.getCurrentPlayerId(google.getApiClient()));
        room_pids = room.getParticipantIds();
        Collections.sort(room_pids);
        server_pid = room_pids.get(0);
        server_role = server_pid.compareTo(my_pid) == 0;
        Log.i("lfl", "server_pid=" + server_pid + ", my_pid=" + my_pid);
    }
    
    @Override
    public void onLeftRoom(int statusCode, String roomId) {
        Log.i("lfl", "GPlusClient::onLeftRoom(" + statusCode + ", " + roomId + ")");
    }
    
    @Override
    public void onJoinedRoom(int statusCode, Room room) {
        if (statusCode != GamesStatusCodes.STATUS_OK) return;
        Log.i("lfl", "GPlusClient.onJoinedRoom(" + room.getRoomId() + ")");
        if (room_id != null) leaveRoom();
        room_id = room.getRoomId();
        
        activity.waiting_activity_result = true;
        activity.startActivityForResult(Games.RealTimeMultiplayer.getWaitingRoomIntent(google.getApiClient(), room, Integer.MAX_VALUE), RC_WAITING_ROOM);
    }

    @Override
    public void onRoomCreated(int statusCode, Room room) {
        if (statusCode != GamesStatusCodes.STATUS_OK) { return; }
        Log.i("lfl", "GPlusClient.onRoomCreated(" + room.getRoomId() + ")");
        if (room_id != null) leaveRoom();
        room_id = room.getRoomId();

        activity.waiting_activity_result = true;
        activity.startActivityForResult(Games.RealTimeMultiplayer.getWaitingRoomIntent(google.getApiClient(), room, Integer.MAX_VALUE), RC_WAITING_ROOM);
    }
    
    @Override
    public void onRealTimeMessageReceived(RealTimeMessage message) {
        int len = message.getMessageData().length;
        // Log.i("lfl", "onRealTimeMessageReceived " + message.getSenderParticipantId() + " " + len);

        if (len > read_buffer.capacity()) {
            Log.e("lfl", len + " > " + read_buffer.capacity());
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
        Log.i("lfl", "GPlusClient.leaveRoom(" + room_id + ")");
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
            // Log.i("lfl", "sendReliable " + room_id + " " + recipientParticipantId);
            if (Games.RealTimeMultiplayer.sendReliableMessage(google.getApiClient(), null, messageData.array(), room_id, recipientParticipantId) < 0)
                Log.e("lfl", "GPlusClient.sendReliable(" + room_id + ", " + recipientParticipantId + ")");
        }

        public void RunUnreliable() {
            if (room_id == null) return;
            // Log.i("lfl", "sendUnreliable " + room_id + " " + recipientParticipantId);
            if (Games.RealTimeMultiplayer.sendUnreliableMessage(google.getApiClient(), messageData.array(), room_id, recipientParticipantId) < 0)
                Log.e("lfl", "GPlusClient.sendUnreliable(" + room_id + ", " + recipientParticipantId + ")");
        }
    }
    
    public void write         (String recipientParticipantId, ByteBuffer messageData) { activity.runOnUiThread(new Write(recipientParticipantId, messageData, false)); }
    public void writeWithRetry(String recipientParticipantId, ByteBuffer messageData) { activity.runOnUiThread(new Write(recipientParticipantId, messageData, true )); }
}

