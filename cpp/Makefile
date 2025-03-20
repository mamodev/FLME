CXX = g++

INCLUDES := $(shell find ./modules -type d | sed 's/^/-I/')

# CXXFLAGS = -std=c++20 -Wall -Wextra $(INCLUDES)
CXXFLAGS = -std=c++20 -g -o0 $(INCLUDES) -luring 

SRC_DIR = modules
BUILD_DIR = build

# MODULE_IFACE_EXT = hpp
MODULE_IMPL_EXT = cpp
MAIN_SRC = main.cpp
TARGET = $(BUILD_DIR)/main

# MODULE_IFACES = $(shell find $(SRC_DIR) -name "*.$(MODULE_IFACE_EXT)")
MODULE_IMPLS = $(shell find $(SRC_DIR) -name "*.$(MODULE_IMPL_EXT)")

# MODULE_IFACE_OBJS = $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%.o,$(MODULE_IFACES))
MODULE_IMPL_OBJS = $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%.o,$(MODULE_IMPLS))
MAIN_OBJ = $(BUILD_DIR)/main.o

# OBJS = $(MODULE_IFACE_OBJS) $(MODULE_IMPL_OBJS) $(MAIN_OBJ)
OBJS = $(MODULE_IMPL_OBJS) $(MAIN_OBJ)



all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BUILD_DIR)


# Test sources and targets
TESTS = $(wildcard tests/*.cpp)
TESTS_TARGETS = $(patsubst tests/%.cpp,build/tests/%,$(TESTS))

TESTS_OBJECTS = $(patsubst tests/%.cpp,build/tests/%.o,$(TESTS))
TEST_BUILD_DIR = build/tests


test: $(TESTS_TARGETS)

# Compilation rules
$(TEST_BUILD_DIR)/% : $(TEST_BUILD_DIR)/%.o $(MODULE_IMPL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_BUILD_DIR)/%.o: tests/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Color and style codes
BOLD=\033[1m
GREEN=\033[0;32m
RED=\033[0;31m
BLUE=\033[0;34m
NC=\033[0m # No Color (reset)


# monitor leak_check full reachable any
# monitor block_list 2
# monitor who_points_at 0x4E489E0
# gdb ./build/tests/defers
# target remote | vgdb --pid=65900
# valgrind --vgdb=yes --vgdb-error=0 ./build/tests/defers

# Common function to run a single test with detailed metrics
define run_test
	test_file=$$(basename $1); \
	test_source=$$(find . -name "$${test_file}.cpp"); \
	printf "$(BLUE)$(BOLD)[Running Test: $$test_file]$(NC) Source: $$test_source\n"; \
	start=$$(date +%s%3N); \
	valgrind -s --error-exitcode=1 --default-suppressions=no --leak-check=full --show-leak-kinds=all --keep-stacktraces=alloc-and-free --trace-children=yes --track-origins=yes --errors-for-leak-kinds=definite --log-file=valgrind_summary.txt $1; \
	exit_code=$$?; \
	end=$$(date +%s%3N); \
	duration=$$((end - start)); \
	\
	in_use_at_exit=$$(grep "in use at exit:" valgrind_summary.txt | awk -F: '{print $$2}'); \
	total_heap_usage=$$(grep "total heap usage:" valgrind_summary.txt | awk -F: '{print $$2}'); \
	definitely_lost=$$(grep "definitely lost:" valgrind_summary.txt | awk -F: '{print $$2}'); \
	still_reachable=$$(grep "still reachable:" valgrind_summary.txt | awk -F: '{print $$2}'); \
	if [ -z "$$mem_usage" ]; then mem_usage=0; fi; \
	\
	if [ $$exit_code -eq 0 ]; then \
		result="$(GREEN)SUCCESS$(NC)"; \
	else \
		result="$(RED)FAIL$(NC)"; \
	fi; \
	\
	printf "\t$(BOLD)[Result]:$(NC) $$result\n"; \
	printf "\t$(BOLD)[Time CPU]:$(NC) $$duration ms\n"; \
	printf "\t$(BOLD)[In Use at Exit]:$(NC) $$in_use_at_exit\n"; \
	printf "\t$(BOLD)[Total Heap Usage]:$(NC) $$total_heap_usage\n"; \
	printf "\t$(BOLD)[Definitely Lost]:$(NC) $$definitely_lost\n"; \
	printf "\t$(BOLD)[Still Reachable]:$(NC) $$still_reachable\n\n"
endef

# rm -f valgrind_summary.txt; 


# Run a single test with metrics
$(notdir $(TESTS_TARGETS)): % : $(TEST_BUILD_DIR)/%.o $(MODULE_IMPL_OBJS)
	@mkdir -p $(BUILD_DIR)/tests
	$(CXX) $(CXXFLAGS) $^ -o build/tests/$@
	@$(call run_test, ./build/tests/$@)


# @$(foreach test,$(TESTS_TARGETS), $(call run_test, $(test)) && ) true
tests: $(TESTS_TARGETS)
	@printf "$(BOLD)$(BLUE)[Running All Tests]$(NC)\n"
	@for test in $(TESTS_TARGETS); do \
		$(call run_test, $$test); \
	done


.PHONY: all clean tests