package xyz.cereshost.common.packet;

import java.io.*;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.Objects;

public final class SocketProperties {
    private final Socket socket;
    private final BufferedOutputStream output;
    private final BufferedInputStream input;

    public SocketProperties(Socket socket, BufferedOutputStream output, BufferedInputStream input) {
        this.socket = socket;
        this.output = output;
        this.input = input;
    }

    public boolean isClosed() {
        return socket.isClosed();
    }

    private final ByteBuffer writeBuffer = ByteBuffer.allocateDirect(4 * 1024 * 1024); // 4MB


    public void close() throws IOException {
        socket.close();
    }


    public ByteBuffer writeBuffer() {
        return writeBuffer;
    }

    public Socket socket() {
        return socket;
    }

    public BufferedOutputStream output() {
        return output;
    }

    public BufferedInputStream input() {
        return input;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (SocketProperties) obj;
        return Objects.equals(this.socket, that.socket) &&
                Objects.equals(this.output, that.output) &&
                Objects.equals(this.input, that.input);
    }

    @Override
    public int hashCode() {
        return Objects.hash(socket, output, input);
    }

    @Override
    public String toString() {
        return "SocketProperties[" +
                "socket=" + socket + ", " +
                "output=" + output + ", " +
                "input=" + input + ']';
    }

}
