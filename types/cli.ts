

type Primitive = "uint8" | "uint16" | "uint32" | "uint64" | "int8" | "int16" | "int32" | "int64" | "float32" | "float64" | `string(${number})`;


type FieldType = Primitive;

type Field = {
    name: string;
    type: FieldType;
};

type Packet = {
    name: string;
    fields: Field[];
};

// CLI USAGE : <CMD> <packet.ts | packet.json> <output-folder>

if(process.argv.length < 4) {
    console.error("Usage: <CMD> <packet.ts | packet.json> <output-folder>");
    process.exit(1);
}

const input = process.argv[2];
const output = process.argv[3];

const inputFile = Bun.file(input);
if(!await inputFile.exists()) {
    console.error("Input file not found");
    process.exit(1);
}

const packets : Packet[] = await inputFile.json() as unknown as Packet[];

function cppPrimitive(type: Primitive): string {
    switch (type) {
        case "uint8":
            return "uint8_t";
        case "uint16":
            return "uint16_t";
        case "uint32":
            return "uint32_t";
        case "uint64":
            return "uint64_t";
        case "int8":
            return "int8_t";
        case "int16":
            return "int16_t";
        case "int32":
            return "int32_t";
        case "int64":
            return "int64_t";
        case "float32":
            return "float";
        case "float64":
            return "double";
        default:
            let match = type.match(/string\((\d+)\)/);
            if (match) {
                return `const char[${match[1]}]`;
            }

            throw new Error(`Unknown primitive type: ${type}`);
    }
}

function sizeOf(f: Field): number | null {
    switch (f.type) {
        case "uint8":
        case "int8":
            return 1;
        case "uint16":
        case "int16":
            return 2;
        case "uint32":
        case "int32":
        case "float32":
            return 4;
        case "uint64":
        case "int64":
        case "float64":
            return 8;
        default:
            let match = f.type.match(/string\((\d+)\)/);
            if (match) {
                return parseInt(match[1]);
            }

            return null;
    }
}

const PACKET_ID_TYPE = "uint32";

function cpp_get_packet_static_size(packet: Packet): [number, boolean] {
    let staticSize = packet.fields.reduce((acc, f) => acc + (sizeOf(f)!), 0);
    let isSizeFullyStatic = packet.fields.every(f => sizeOf(f) !== null);
    return [staticSize, isSizeFullyStatic];
}


async function generateCPP(packets: Packet[]) {

    const hpp = Bun.file(output + "/protocol.hpp");
    const cpp = Bun.file(output + "/protocol.cpp");
    const handlers = Bun.file(output + "/handlers.hpp");

    let handlersCode = `
    #pragma once
    #include "protocol.hpp"
    #include "asyncc.hpp"

    namespace protocol {
    `;  

    let cppCode = `
    #include "protocol.hpp"
    #include "handlers.hpp"
    namespace protocol {
    `;

    let hppCode = `
    #pragma once
    #include <variant>
    #include "result_types.hpp"

    namespace protocol {
    #define PACKET_ID_SIZE ${sizeOf({name: "id", type: PACKET_ID_TYPE})}
    `

    for (const packet of packets) {
        let staticSize = packet.fields.reduce((acc, f) => acc + (sizeOf(f)!), 0);
        let isSizeFullyStatic = packet.fields.every(f => sizeOf(f) !== null);

        hppCode += `#pragma pack(push, 1)\n`;
        hppCode += `struct ${packet.name} {\n`;
        if(staticSize !== null) {
            hppCode += `    //this rapresent ${isSizeFullyStatic ? "total" : 'minimum'} size of the packet\n`;
            hppCode += `    static constexpr size_t size = ${staticSize};\n`;
        }


        for (const field of packet.fields) {
            hppCode += `    ${cppPrimitive(field.type)} ${field.name};\n`;
        }


        hppCode += `};\n`;
        hppCode += `#pragma pack(pop)\n\n`;

        handlersCode += `Task<Res<void>> ${packet.name}Handler(Conn& conn, ${packet.name}& packet);\n`;
    }

    hppCode += `Task<Res<void>> handleRawPacket(Conn& conn, void* data, size_t size) {\n`;


    //a raw packet is composed by packet ID and the Packet itself
    cppCode += `Task<Res<void>> handleRawPacket(Conn& conn, void* data, size_t size) {\n`;
    cppCode += `    if(size < PACKET_ID_SIZE)\n`;
    cppCode += `        return Error("Invalid packet size, expected at least " PACKET_ID_SIZE " bytes, for the packet ID");\n`;
    cppCode += '\n';
    cppCode += `    ${PACKET_ID_TYPE} id = *(${PACKET_ID_TYPE}*)data;\n`;
    cppCode += '\n';
    cppCode += `    switch(id) {\n`;
    
    for (let i = 0; i < packets.length; i++) {
        const [staticSize, isSizeFullyStatic] = cpp_get_packet_static_size(packets[i]);

        cppCode += `        case ${i}:\n`;

        cppCode += `            ${packets[i].name} packet;\n`;
        if (isSizeFullyStatic) {
            cppCode += `            if(size != ${packets[i].name}::size)\n`;
            cppCode += `               return Error("Invalid packet size, expected exactly " ${packets[i].name}::size " bytes, for the packet ${packets[i].name}");\n`;
            cppCode += '\n';
            cppCode += `            //this will work only if the system uses little endian\n`;
            cppCode += `            memcpy(&packet, data + PACKET_ID_SIZE, ${packets[i].name}::size);\n`;

        } else {
            cppCode += `            if(size < ${packets[i].name}::size)\n`;
            cppCode += `                return Error("Invalid packet size, expected at least " ${packets[i].name}::size " bytes, for the packet ${packets[i].name}");\n`;
            cppCode += '\n';
            cppCode += `            //this will work only if the system uses little endian\n`;

            // copy the static part of the packet
            cppCode += `            memcpy(&packet, data + PACKET_ID_SIZE, ${staticSize});\n`;
            //TODO: copy the dynamic part of the packet
        }

        cppCode += `            releaseBuffer(data, size);\n`;
        cppCode += '\n';
        cppCode += `            ${packets[i].name}Handler(conn, packet);\n`;
        cppCode += `            return Res<void>(std::nullopt);\n`;
    }
    
    cppCode += `
        default:
            return Error("Unknown packet id: " + std::to_string(id));
    }
    `;


    hppCode += `}\n`;
    cppCode += `}\n`;
    handlersCode += `}\n`;
    hpp.write(hppCode);
    cpp.write(cppCode);
    handlers.write(handlersCode);
}

await generateCPP(packets);