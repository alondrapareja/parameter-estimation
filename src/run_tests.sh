#Runs all the tests in the tests directory
python3 -m unittest discover -s tests -p "*.py" > result.log

#Prints test results to console
cat result.log

#Notifies test have been ran
echo "The tests have been ran!"
