package xyz.cereshost.common.packet;

import com.google.gson.reflect.TypeToken;
import lombok.Getter;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

@SuppressWarnings("unchecked")
@Getter
public abstract class PacketListener<T extends Packet> {

    private final Class<T> clazz;

    public PacketListener(){
        Type type = ((ParameterizedType) getClass().getGenericSuperclass()).getActualTypeArguments()[0];
        this.clazz = (Class<T>) TypeToken.get(type).getRawType();
        BasePacketHandler.addListener(this);
    }

    public void receivePacket(Packet packet){
        onReceive((T) packet);
    }

    protected abstract void onReceive(T packet);
}
