NVCC ?= nvcc
ARCH ?= -arch=sm_60

SRC  = main.cu
BIN  = binary_tree_gpu

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(ARCH) -O3 -std=c++14 -o $@ $<

test: $(BIN)
	@echo "[TEST] Running basic correctness tests..."
	@./$(BIN) 65536 A
	@./$(BIN) 262144 A
	@./$(BIN) 1048576 A
	@echo "[TEST] Finished. Check that all runs report 'CPU and GPU results match'."

benchmark: $(BIN)
	@echo "[BENCHMARK] Running performance benchmarks..."
	@./$(BIN) 65536 A
	@./$(BIN) 262144 A
	@./$(BIN) 1048576 A
	@echo "[BENCHMARK] Finished. Record CPU/GPU times and throughput from the output."

clean:
	rm -f $(BIN)

.PHONY: all clean
