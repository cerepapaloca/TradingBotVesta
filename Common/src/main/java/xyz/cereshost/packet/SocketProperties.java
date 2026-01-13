package xyz.cereshost.packet;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

public record SocketProperties(Socket socket, DataOutputStream output, DataInputStream input) {
    public boolean isClosed(){
        return socket.isClosed();
    }

    public void close() throws IOException {
        socket.close();
    }
}
