# MAKEFILE
CC = g++


#FLAGS=$(shell pkg-config --static --libs --cflags seastar/build/release/seastar.pc)
FLAGS=-luring

OBJ_DIR = obj
BIN_DIR = bin
SRC_DIR = src

$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

all: $(BIN_DIR)/main

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(BIN_DIR)/main
	$(BIN_DIR)/main -m8G

$(BIN_DIR)/main: $(SRC_DIR)/main.cpp
	@$(CC) $(SRC_DIR)/main.cpp -o $(BIN_DIR)/main $(FLAGS)


