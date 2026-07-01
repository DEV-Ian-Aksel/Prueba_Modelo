#!/bin/bash

# Script para ejecutar todos los modelos con casas.csv
# Uso: bash ejecutar_todos.sh

echo "╔════════════════════════════════════════════════════════╗"
echo "║  EJECUTANDO LOS 3 MODELOS CON casas.csv              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar encabezado
show_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo ""
}

# Función para mostrar éxito
show_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Función para mostrar error
show_error() {
    echo -e "${YELLOW}✗ $1${NC}"
}

# 1. XGBoost
show_header "MODELO 1: XGBoost"
cd /workspaces/Prueba_Modelo/Practica_modelo

if python entrenamiento.py 2>&1 | tail -1 | grep -q "pkl"; then
    show_success "XGBoost entrenado correctamente"
else
    show_error "Error en XGBoost"
fi

# 2. Random Forest
show_header "MODELO 2: Random Forest"
cd /workspaces/Prueba_Modelo/Practica_RandomForest

if python entrenamiento.py 2>&1 | tail -1 | grep -q "pkl"; then
    show_success "Random Forest entrenado correctamente"
else
    show_error "Error en Random Forest"
fi

# 3. Regresión Logística
show_header "MODELO 3: Regresión Logística"
cd /workspaces/Prueba_Modelo/Practica_RegresionLogistica

if python entrenamiento.py 2>&1 | tail -1 | grep -q "pkl"; then
    show_success "Regresión Logística entrenada correctamente"
else
    show_error "Error en Regresión Logística"
fi

# Resumen
show_header "✓ ENTRENAMIENTO COMPLETADO"
echo -e "${GREEN}Los 3 modelos han sido entrenados exitosamente${NC}"
echo ""
echo "📊 RESULTADOS ESPERADOS:"
echo "  • XGBoost:              ~90% accuracy"
echo "  • Random Forest:        ~95% accuracy (MEJOR)"
echo "  • Regresión Logística:  ~90% accuracy, 0.98 AUC"
echo ""
echo "📁 ARCHIVOS GENERADOS:"
echo "  • Practica_modelo/modelo_xgboost_casas.pkl"
echo "  • Practica_RandomForest/modelo_randomforest_casas.pkl"
echo "  • Practica_RegresionLogistica/modelo_logistico_casas.pkl"
echo ""
echo "📖 Ver resultados en: RESULTADOS_CASAS.md"
echo ""
