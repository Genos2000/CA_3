all: data/input_16.in data/input_2048.in data/input_4096.in data/input_8192.in data/input_16384.in rmm

rmm: main.cpp header/single_thread.h header/multi_thread.h
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2

data/generate: data/generate.cpp
	g++ ./data/generate.cpp -o ./data/generate

data/input_16.in: data/generate
	./data/generate 16 

data/input_4096.in: data/generate
	./data/generate 4096 

data/input_8192.in: data/generate
	./data/generate 8192 

data/input_16384.in: data/generate
	./data/generate 16384 

run: data/input_16.in data/input_2048.in data/input_4096.in data/input_8192.in data/input_16384.in rmm
	./rmm data/input_16.in
	./rmm data/input_4096.in
	./rmm data/input_8192.in
	./rmm data/input_16384.in

8: data/input_8.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_8.in

16: data/input_16.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_16.in

32: data/input_32.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_32.in

1024: data/input_1024.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_1024.in

2048: data/input_2048.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_2048.in

4096: data/input_4096.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_4096.in

8k: data/input_8192.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_8192.in

16k: data/input_16384.in rmm
	rm rmm
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2
	./rmm data/input_16384.in

clean:
	rm rmm
