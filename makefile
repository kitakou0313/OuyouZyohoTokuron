.PHONY: exec1
exec1:
	@python main.py

.PHONY: exec2
exec2:
	@python main2.py

.PHONY: exec3
exec3:
	@python main3.py

.PHONY: logExec1
logExec1:
	@python main.py > logs/ex1.txt

.PHONY: logExec2
logExec2:
	@python main2.py > logs/ex2.txt

.PHONY: logExec3
logExec3:
	@python main3.py > logs/ex3.txt