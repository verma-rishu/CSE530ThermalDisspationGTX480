#!/bin/bash
#BlockSize=512;
#kernel_name="matmul"

print_usage(){
	echo "Invalid number of parameters"
	echo "Usage: source ./runProject.sh <block_dimenstion> <kernelname>"
	echo "block_dimenstion > 0"
	echo "kernelname should be valid file in the directory with out cuda extention! eg: For matrixMultiplicaiton.cu only pass matrixMultiplicaiton"
	exit 1
}

if (( $# < 2 )); then
	print_usage
fi

if (( $1 <= 0 )); then
	print_usage
fi

if (( $1 <= 0 )); then
	print_usage
fi

if [[ -f ./kernel/$2.cu ]]; then
	if [ `ls ./kernel/$2.cu 2> /dev/null| wc -l ` -gt 0 ]; then
		echo "Running Kernel!"
	else
		echo "Cuda file not present in the Kernel folder!"
    	print_usage
	fi

else
    echo "$2 is not a valid file name"
    exit 1
fi

BlockSize=$1;
kernel_name=$2;

# file system setup
mkdir -p generatedFiles$BlockSize
mkdir -p ./generatedFiles$BlockSize/app_cuda_versions
mkdir -p ./generatedFiles$BlockSize/cuobjdumps
mkdir -p ./generatedFiles$BlockSize/gpgpusim_metric_trace_report_logs
mkdir -p ./generatedFiles$BlockSize/gpgpusim_power_report_logs
mkdir -p ./generatedFiles$BlockSize/gpgpusim_power_trace_report_logs
mkdir -p ./generatedFiles$BlockSize/gpgpusim_steady_state_tracking_report_logs
mkdir -p ./generatedFiles$BlockSize/ptx_files/
mkdir -p ./generatedFiles$BlockSize/ptxas_files/
mkdir -p ./generatedFiles$BlockSize/ptxas_files/
mkdir -p ./generatedFiles$BlockSize/gpgpu_inst_stats/
mkdir -p ./generatedFiles$BlockSize/kernel_output/
mkdir -p ./generatedFiles$BlockSize/ptrace/
mkdir -p ./generatedFiles$BlockSize/hotspotOutput/

# Compiling the kernel
	echo "Compiling the Matrix Multiplication kernel..."
	nvcc ./kernel/${kernel_name}.cu -lcudart -o ${kernel_name}
	echo "Compilation done"

	libcudartLocation="$(/home/grads/rfv5129/CSE530Project/gpgpu-sim_distribution/lib/gcc-7.3.0/cuda-11080/release/libcudart.so.11.0)"

# Checking if the correct libcudart lib is linked to the kernel executable
	if [ `ldd ${kernel_name} | grep "libcudart.so.11.0 => ${libcudart_so_location}" | wc -l ` -ne 1 ]; then
		echo "Compilation/Build of gpgpu-sim wasn't successful, source the setup_environment from the gpgpu-sim"
		return
	fi

# Executing the Matrix Multiplication kernel
	echo "Executing the Matrix Multiplication kernel..."
	mat_mul_kernel_resultfile="${kernel_name}${BlockSize}_$(date "+%Y_%m_%d-%H_%M_%S").txt"
	./$kernel_name $BlockSize -power_simulation_enabled 1 -gpuwattch_xml_file "gpuwattch_gtx480.xml" > $mat_mul_kernel_resultfile
	mv $mat_mul_kernel_resultfile generatedFiles$BlockSize/kernel_output/
	echo "Execution done"

#moving files to generatedFiles
echo "Moving generated files to the generatedFiles folder"

if [ `ls gpgpu_inst_stats*.txt 2> /dev/null | wc -l ` -gt 0 ]; then
	# fullFileName=$(find . -type f -name "gpgpu_inst_stats*.txt" -printf "%f\n")#$(find -type f -name 'gpgpu_inst_stats*'  -exec basename {} \');
	# filename=${fullFileName%.*}
	# fileextension=${fullFileName##*.}
	# newFileName="${filename}${BlockSize}.${fileextension}"
	# mv ./"$fullFileName" ./"$newFileName"
	mv gpgpu_inst_stats*.txt generatedFiles$BlockSize/gpgpu_inst_stats/
fi

if [ `ls _app_cuda_version_* 2> /dev/null| wc -l ` -gt 0 ]; then
	mv _app_cuda_version_* generatedFiles$BlockSize/app_cuda_versions/
fi

if [ `ls _cuobjdump_list_ptx_* 2> /dev/null| wc -l ` -gt 0 ]; then
	mv _cuobjdump_list_ptx_* generatedFiles$BlockSize/cuobjdumps/
fi

if [ `ls *.ptx 2> /dev/null| wc -l ` -gt 0 ]; then
	mv *.ptx generatedFiles$BlockSize/ptx_files/
fi

if [ `ls *.ptxas 2> /dev/null | wc -l ` -gt 0 ]; then
	mv *.ptxas generatedFiles$BlockSize/ptxas_files/
fi

if [ `ls gpgpusim_power_report__* 2> /dev/null | wc -l ` -gt 0 ]; then
	mv gpgpusim_power_report__* generatedFiles$BlockSize/gpgpusim_power_report_logs/
fi

if [ `ls gpgpusim_metric_trace_report__* 2> /dev/null | wc -l ` -gt 0 ]; then
	mv gpgpusim_metric_trace_report__* generatedFiles$BlockSize/gpgpusim_metric_trace_report_logs/
fi

if [ `ls gpgpusim_power_trace_report__* 2> /dev/null | wc -l ` -gt 0 ]; then
	mv gpgpusim_power_trace_report__* generatedFiles$BlockSize/gpgpusim_power_trace_report_logs/
fi

if [ `ls gpgpusim_steady_state_tracking_report__* 2> /dev/null | wc -l ` -gt 0 ]; then
	mv gpgpusim_steady_state_tracking_report__* generatedFiles$BlockSize/gpgpusim_steady_state_tracking_report_logs/
fi

if [ `ls *_out 2> /dev/null | wc -l ` -gt 0 ]; then
	mv *_out generatedFiles$BlockSize/kernel_output/
fi

if [ `ls *_RHS 2> /dev/null | wc -l ` -gt 0 ]; then
	mv *_RHS generatedFiles$BlockSize/kernel_output/
fi

if [ `ls *_LHS 2> /dev/null | wc -l ` -gt 0 ]; then
	mv *_LHS generatedFiles$BlockSize/kernel_output/
fi

echo "Done moving files!!"

# Generating the ptrace file from the log report
echo "Generating the ptrace file....."
logFile=$(find ./generatedFiles$BlockSize/gpgpusim_power_report_logs/ -type f -name "*.log" -printf "%f")
logFilePath="./generatedFiles$BlockSize/gpgpusim_power_report_logs/$logFile"
outputPtrace=output$BlockSize.ptrace

	python3 logToPtrace.py $logFilePath $outputPtrace

	if [ `ls -1 $outputPtrace 2>/dev/null | wc -l ` -ne 1 ]; then
		echo "Generation of the ptrace file failed"
		return
	fi
echo "Generation of ptrace done"

# Running HotSpot
echo "Running hotspot..."

# Remove results from previous simulations
rm -f *.init
rm -f outputs$BlockSize/*

# Create outputs directory if it doesn't exist
mkdir outputs$BlockSize

outputDir=outputs$BlockSize

# in the LCF instead of via the command line
../HotSpot/hotspot -c GTX480Fermi.config -p $outputPtrace -grid_layer_file GTX480Fermi.lcf -materials_file GTX480Fermi.materials -model_type grid -detailed_3D on -steady_file $outputDir/GTX480Fermi.steady -grid_steady_file $outputDir/GTX480Fermi.grid.steady

# Copy steady-state results over to initial temperatures
cp $outputDir/GTX480Fermi.steady GTX480Fermi.init

# Transient simulation
../HotSpot/hotspot -c GTX480Fermi.config -p $outputPtrace -grid_layer_file GTX480Fermi.lcf -materials_file GTX480Fermi.materials -model_type grid -detailed_3D on -o $outputDir/GTX480Fermi.ttrace -grid_transient_file $outputDir/GTX480Fermi.grid.ttrace

# Visualize Heat Map of Layer 2- Silicon with Perl and with Python script
python3 ../HotSpot/scripts/split_grid_steady.py $outputDir/GTX480Fermi.grid.steady 4 64 64
python3 ../HotSpot/scripts/grid_thermal_map.py floorplan2.flp $outputDir/GTX480Fermi_layer2.grid.steady 64 64 $outputDir/layer2.png
../HotSpot/scripts/grid_thermal_map.pl floorplan2.flp $outputDir/GTX480Fermi_layer2.grid.steady 64 64 > $outputDir/layer2.svg

if [ `ls $outputPtrace 2> /dev/null | wc -l ` -gt 0 ]; then
	mv $outputPtrace generatedFiles$BlockSize/ptrace/
fi
if [ `ls $outputDir 2> /dev/null | wc -l ` -gt 0 ]; then
	mv $outputDir generatedFiles$BlockSize/hotspotOutput/
fi