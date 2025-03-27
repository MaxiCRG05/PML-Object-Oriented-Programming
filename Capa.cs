using System;
using System.Collections.Generic;
using System.Linq;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Capa para manejar las capas del PML.
	/// Aquí se hace la retropropagación, la propagación y el cálculo de la función de activación y su derivada.
	/// También se manejan las neuronas.
	/// </summary>
	public class Capa
	{
		/// <summary>
		/// Neuronas de la capa.
		/// </summary>
		public List<Neurona> neuronas = new List<Neurona>();

		/// <summary>
		/// Número de neuronas en la capa siguiente.
		/// Número de neuronas en la capa actual.
		/// </summary>
		public int numNeuronasCapaSiguiente, numNeuronas;

		/// <summary>
		/// Establece el tipo de capa que puede ser de entrada, oculta o salida.
		/// </summary>
		public TipoCapa Tipo { get; private set; }

		/// <summary>
		/// Enum para definir el tipo de capa.
		/// </summary>
		public enum TipoCapa
		{
			Entrada,
			Oculta,
			Salida
		}

		/// <summary>
		/// Constructor de la clase Capa.
		/// </summary>
		/// <param name="numeroNeuronas">Número de neuronas en la capa.</param>
		/// <param name="numeroNeuronasCapaSiguiente">Número de neuronas en la capa siguiente.</param>
		/// <param name="tipo">Tipo de capa (Entrada, Oculta o Salida).</param>
		public Capa(int numeroNeuronas, int numeroNeuronasCapaSiguiente, TipoCapa tipo)
		{
			this.Tipo = tipo;
			this.numNeuronas = numeroNeuronas;
			this.numNeuronasCapaSiguiente = numeroNeuronasCapaSiguiente;

			agregarNeuronas(numeroNeuronas, numeroNeuronasCapaSiguiente);
		}

		/// <summary>
		/// Agrega neuronas a la capa correspondiente.
		/// </summary>
		/// <param name="numNeuronas">Número de neuronas en la capa actual.</param>
		/// <param name="numNeuronasCapaSiguiente">Número de neuronas en la capa siguiente.</param>
		public void agregarNeuronas(int numNeuronas, int numNeuronasCapaSiguiente)
		{
			for (int i = 0; i < numNeuronas; i++)
			{
				string nombre = Tipo == TipoCapa.Entrada ? $"Neurona_Entrada_{i +1}" :
								Tipo == TipoCapa.Oculta ? $"Neurona_Oculta_{i + 1}" :
								$"Neurona_Salida_{i + 1}";

				neuronas.Add(new Neurona(nombre, numNeuronas, numNeuronasCapaSiguiente, Tipo));
			}
		}


		/// <summary>
		/// Calcula la activación en la capa de entrada.
		/// </summary>
		/// <param name="x">Patrones.</param>
		/// <param name="y">Posición.</param>
		public void calcularActivacion(double[] x)
		{
			for(int i = 0; i < neuronas.Count; i++)
			{
				neuronas[i].a = x[i];
			}
		}

		/// <summary>
		/// Calcula la activación en capas ocultas y de salida.
		/// </summary>
		public void calcularActivacion()
		{
			for (int i = 0; i < neuronas.Count; i++)
			{
				double suma = 0;
				for(int j = 0; j < neuronas[i].neuronasAnteriores.Count; j++)
				{
					suma += neuronas[i].neuronasAnteriores[j].a * neuronas[i].neuronasAnteriores[j].w[i];
				}
				suma += neuronas[i].b;
				neuronas[i].a = funcionActivacion(suma);
			}
		}

		/// <summary>
		/// Calcula los errores de la capa de salida.
		/// </summary>
		/// <param name="s">Patrones esperados</param>
		public void calcularError(double[] s)
		{
			for (int i = 0; i < neuronas.Count; i++)
			{
				neuronas[i].delta = (s[i] - neuronas[i].a) * funcionDeActivacionDerivada(neuronas[i].a);
			}
		}

		/// <summary>
		/// Calcula los valore delta de la última capa oculta, hasta la primera capa oculta.
		/// </summary>
		public void calcularError()
		{
			for (int j = 0; j < neuronas.Count; j++)
			{
				double suma = 0;
				foreach (var neuronaSiguiente in neuronas[j].neuronasSiguientes)
				{
					suma += neuronaSiguiente.delta * neuronaSiguiente.w[j];
				}
				neuronas[j].delta = suma * funcionDeActivacionDerivada(neuronas[j].a);
			}
		}

		public void actualizarBiasPesos()
		{
			for (int j = 0; j < neuronas.Count; j++)
			{
				for (int i = 0; i < neuronas[j].neuronasAnteriores.Count; i++)
				{
					neuronas[j].w[i] += VariablesGlobales.TasaAprendizaje * neuronas[j].delta * neuronas[j].neuronasAnteriores[i].a;
				}
				neuronas[j].b += VariablesGlobales.TasaAprendizaje * neuronas[j].delta;
			}
		}

		/// <summary>
		/// Método para la función de activación: funcion RELU, LEAKY RELU Y SIGMOIDE
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>Regresa el valor de entrada procesada con base a la función de activación.</returns>
		private double funcionActivacion(double x)
		{
			//SIGMOIDE
			//return 1 / (1 + Math.Exp(-x));

			//RELU
			//return Math.Max(0,x);

			// Leaky ReLU
			return x > 0 ? x : 0.01 * x;

			//ELU
			//return x > 0 ? x : 0.01 * (Math.Exp(x) - 1);
		}

		/// <summary>
		/// Método para la función de activación derivada: funcion RELU, LEAKY RELU Y SIGMOIDE DERIVADA
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>La entrada ya procesada por la función de activación derivada.</returns>
		private double funcionDeActivacionDerivada(double x)
		{
			//SIGMOIDE
			//return x * (1 - x);

			//RELU
			//return x > 0 ? 1 : 0;

			//Leaky ReLU
			return x >= 0 ? 1 : 0.01;

			//ELU
			//return x > 0 ? 1 : 0.01 * Math.Exp(x);
		}
	}
}