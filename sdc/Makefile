
all: main 

main: main.cpp async.hpp
	g++ -g main.cpp -o main -std=c++23 -luring -I./

# echo: echo.cpp async.hpp
# 	g++ echo.cpp -o echo -std=c++20 -luring -I./

run: main
	./main

clean:
	rm -f main

.PHONY: run clean
