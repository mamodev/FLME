package main

import (
	"encoding/binary"
	"fmt"
	"net"
)

type Model struct {
	ID   int
	Data []byte
}

type Packet struct {
	ID   int
	Data []byte
}

type Aggregator interface {
	GetModel(id int) Model
	PutModel(model Model)
	RegisterOnModelUpdate(func(id int, agg Aggregator))
}

type FedAvgAggregator struct {
	models          map[int]Model
	latest_model_id int
	updates         []Model
	listeners       []func(id int, agg Aggregator)
}

func (agg *FedAvgAggregator) GetModel(id int) Model {
	panic("Not implemented")
}

func (agg *FedAvgAggregator) PutModel(model Model) {
	panic("Not implemented")
}

func (agg *FedAvgAggregator) RegisterOnModelUpdate(f func(id int, agg Aggregator)) {
	panic("Not implemented")
}

func handleGetModel(conn net.Conn, packet Packet, agg Aggregator) {
	modelID := binary.BigEndian.Uint16(packet.Data)
	model := agg.GetModel(int(modelID))

	// send model to client
	conn.Write([]byte{0, 0, 0, 0}) // packet size
	conn.Write([]byte{0, 2})        // packet ID
	conn.Write([]byte{byte(model.ID >> 8), byte(model.ID & 0xff})
	conn.Write(model.Data)
}

func handleSendModel(conn net.Conn, packet Packet, agg Aggregator) {
}

func handlePacket(conn net.Conn, packet Packet, agg Aggregator) {
	fmt.Printf("Received packet with ID %d\n", packet.ID)

	switch packet.ID {
	case 1:

	default:
		fmt.Println("Unknown packet ID")
	}
}

func handleConnection(conn net.Conn, agg Aggregator) {
	defer conn.Close()

	// read first uint32 as packet size in network byte order
	var size uint32
	err := binary.Read(conn, binary.BigEndian, &size)
	if err != nil {
		panic(err)
	}

	if size < 2 {
		fmt.Println("Invalid packet size, must be at least 2 bytes (packet ID)")
		return
	}

	// read the rest of the packet
	data := make([]byte, size)
	for read := 0; read < int(size); {
		n, err := conn.Read(data[read:])
		if err != nil {
			fmt.Println("Error reading data from connection")
			return
		}
		read += n
	}

	// read packet ID uint16
	packetID := binary.BigEndian.Uint16(data[:2])

	// create packet struct
	packet := Packet{
		ID:   int(packetID),
		Data: data[2:],
	}

	handlePacket(conn, packet, agg)
}

func server(port int) {
	litener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		panic(err)
	}

	agg := &FedAvgAggregator{}

	for {
		conn, err := litener.Accept()
		if err != nil {
			panic(err)
		}

		go handleConnection(conn, agg)
	}
}

func main() {
	port := 8080
	server(port)

}
