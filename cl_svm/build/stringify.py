import sys

if (__name__ == "__main__"):
	data = sys.stdin.readlines()
	outputData = []
	for line in data:
		outputData.append( "\"" + line.strip("\n") + "\\n\" \\" );
	for line in outputData:
		print(line)
	print( "\"\"" )