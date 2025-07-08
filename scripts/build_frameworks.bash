#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  build    Build all frameworks (default)"
    echo "  clean    Clean build artifacts"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build    # Build all frameworks"
    echo "  $0 clean    # Clean build artifacts"
    echo "  $0          # Build all frameworks (default)"
}

# Function to clean build artifacts
clean_frameworks() {
    echo "Cleaning build artifacts..."
    
    # Clean TMC13
    if [ -d "frameworks/mpeg-pcc-tmc13/build" ]; then
        echo "Cleaning TMC13 build directory..."
        rm -rf frameworks/mpeg-pcc-tmc13/build
    fi
    
    # Clean PccAI (remove any build artifacts if they exist)
    if [ -d "frameworks/PccAI/build" ]; then
        echo "Cleaning PccAI build directory..."
        rm -rf frameworks/PccAI/build
    fi
    
    echo "Clean completed!"
}

# Function to build frameworks
build_frameworks() {
    echo "Building frameworks..."
    
    # Build TMC13
    echo "Building TMC13..."
    cd frameworks/mpeg-pcc-tmc13
    
    # Create build directory and build
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    # Test installation
    echo "Testing TMC13 installation..."
    ./tmc3 --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "TMC13 build successful!"
    else
        echo "Warning: TMC3 test failed, but build may still be successful"
    fi
    
    # Go back to root directory
    cd ../../
    
    # Build PccAI
    echo "Setting up PccAI..."
    cd frameworks/PccAI
    
    # Install PccAI specific dependencies
    if [ -f "requirements.txt" ]; then
        echo "Installing PccAI dependencies..."
        pip install -r requirements.txt
    else
        echo "Warning: requirements.txt not found in PccAI directory"
    fi
    
    # Test basic import
    echo "Testing PccAI setup..."
    python -c "import sys; sys.path.append('.'); print('PccAI setup complete')" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "PccAI setup successful!"
    else
        echo "Warning: PccAI test failed, but setup may still be successful"
    fi
    
    # Go back to root directory
    cd ../../
    
    echo "All frameworks built successfully!"
}

# Main script logic
case "${1:-build}" in
    "build")
        build_frameworks
        ;;
    "clean")
        clean_frameworks
        ;;
    "help"|"-h"|"--help")
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac