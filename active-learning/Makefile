#http://www.gnu.org/software/make/manual/make.html

#
# This is here to prevent Make from deleting secondary files.
#
.SECONDARY:

EXE=trainer.out classifier.out ranker.out
CXX=g++
CXXFLAGS=-Wall -ggdb -Idlib-18.9

all: dirs $(addprefix bin/, $(EXE)) tags

debug: CXXFLAGS += -DDEBUG -g
debug: all

dirs:
	mkdir -p obj
	mkdir -p bin

tags: *.cpp
	ctags *.cpp

obj/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.out: obj/%.o $(OBJ)
	$(CXX) $(CXXFLAGS) $< $(OBJ) $(LDFLAGS) -o $@

clean:
	rm -f bin/*
	rm -f obj/*
	rm -f tags
