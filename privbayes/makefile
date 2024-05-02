UNAME := $(shell uname)

all:
	$(info UNAME is $(UNAME))
	g++ -O3 --std=c++14 code.degree/main_marginal.cpp code.degree/methods.cpp code.degree/noise.cpp code.degree/printer.cpp code.degree/table.cpp code.degree/translator.cpp -o privBayesDegree.bin
	if [ "$(UNAME)" = "Linux" ]; then \
		g++ -O3 --std=c++14 code.theta/main_marginal.cpp code.theta/methods.cpp code.theta/noise.cpp code.theta/printer.cpp code.theta/table.cpp code.theta/translator.cpp -o privBayesTheta.bin -lboost_random; \
	else \
		g++ -O3 --std=c++14 code.theta/main_marginal.cpp code.theta/methods.cpp code.theta/noise.cpp code.theta/printer.cpp code.theta/table.cpp code.theta/translator.cpp -o privBayesTheta.bin -lboost_random-mt; \
	fi
