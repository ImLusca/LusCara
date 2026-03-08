# Variáveis de Compilação
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra

# Nome do Executável Final
TARGET   := facenet_embedder

# Diretórios
SRC_DIR  := src
INC_DIR  := include
OBJ_DIR  := obj
MODELS_DIR := models

# Dependências Externas (ONNX Runtime e OpenCV)
ONNX_DIR := ../third_party/onnxruntime/onnxruntime-linux-x64-1.17.0
ONNX_INC := $(ONNX_DIR)/include
ONNX_LIB := $(ONNX_DIR)/lib

# Flags para o Compilador (Includes)
# Adicionamos a sua pasta include, a do ONNX e as do OpenCV
CXXFLAGS += -I$(INC_DIR) -I$(ONNX_INC) $(shell pkg-config --cflags opencv4)

# Flags para o Linker (Bibliotecas)
LDFLAGS  := -L$(ONNX_LIB) -lonnxruntime $(shell pkg-config --libs opencv4)
# RPATH permite que o binário encontre a .so do ONNX na pasta lib do projeto
LDFLAGS  += -Wl,-rpath,'$$ORIGIN/lib'

# Localização das fontes e objetos
# Busca todos os .cpp dentro de src/
SRCS     := $(shell find $(SRC_DIR) -name "*.cpp")
# Transforma a lista de .cpp em uma lista de .o dentro da pasta obj/
OBJS     := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# --- REGRAS DE BUILD ---

all: $(TARGET)

# Regra para o executável final
$(TARGET): $(OBJS)
	@echo "Linking: $@"
	@$(CXX) $(OBJS) $(LDFLAGS) -o $@

# Regra genérica para compilar arquivos .cpp em .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Limpeza
clean:
	@echo "Cleaning up..."
	@rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean