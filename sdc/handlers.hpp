
    #pragma once
    #include "protocol.hpp"
    #include "asyncc.hpp"

    namespace protocol {
    Task<Res<void>> AuthPacketHandler(Conn& conn, AuthPacket& packet);
Task<Res<void>> PushModelHandler(Conn& conn, PushModel& packet);
}
