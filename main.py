#from src.L import *
from src.automatas.automata_learning import Learner as AutomataLearner
from experiments.performance import measure_automata_performance_in_functions, measure_transducer_performance_in_datasets

def f(xs):
	res = []
	last = '0'
	for x in xs:
		if x == '1':
			last = '0' if last == '1' else '1'
		res.append(last)
	return res


if __name__ == "__main__":
	#measure_transducer_performance_in_datasets()


	#learner = TransducerLearner(16, alphabet_in_6, alphabet_out_6)
	#transducer = learner.learn_from_dataset(xs6, ys6, run_n=10, verbose=0)
	#print(f"    Square Error Sum: {transducer.error_square(xs6, ys6)}\n")


	#learner = AutomataLearner(2, ['0', '1'])
	#xs = ['011101', '01010', '111111', '11']
	#ys = ['010101', '01010', '010101', '01']
	#automata = learner.learn_from_dataset(xs, ys, run_n=256, verbose=0)
	#print(f"    Square Error Sum: {automata.error_square(xs, ys)}\n")
	#automata_s = automata.to_state_automata()
	#automata_s.print()


	res = measure_automata_performance_in_functions(pr=0.75, le=10, run_n=2000)
	print(res)

	import matplotlib.pyplot as plt
	import numpy as np

	automatas, train_errors, test_errors, times = res

	indices = np.array([i for i in range(max(len(train_errors), len(test_errors)))])

	# El ancho de cada barra
	width = 0.35

	# Crear el gráfico de barras
	fig, ax = plt.subplots()


	# Gráficos de barras para cada lista
	bar1 = ax.bar(indices - width / 2, train_errors, width, label='train')
	bar2 = ax.bar(indices + width / 2, test_errors, width, label='test')

	# Añadir etiquetas, título y leyenda
	ax.set_xlabel('Índice')
	ax.set_ylabel('Valores')
	ax.set_title('Bar plot conjunto de dos listas')
	ax.set_xticks(indices)
	ax.set_xticklabels([f'{i}' for i in indices])
	ax.legend()

	# Mostrar el gráfico
	plt.show()

