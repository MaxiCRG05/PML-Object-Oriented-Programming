﻿using System;

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
		public Neurona[] Neuronas { get; private set; }
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
			Tipo = tipo;
			Neuronas = new Neurona[numeroNeuronas];

			if (tipo == TipoCapa.Entrada)
			{
				for (int i = 0; i < numeroNeuronas; i++)
				{
					Neuronas[i] = new Neurona($"Neurona_{i}", 0);
				}
			}
			else
			{
				for (int i = 0; i < numeroNeuronas; i++)
				{
					Neuronas[i] = new Neurona($"Neurona_{i}", numeroNeuronasCapaSiguiente);
				}
			}
		}

		/// <summary>
		/// Realiza la propagación hacia adelante (forward propagation).
		/// </summary>
		/// <param name="entradas">Entradas de la capa.</param>
		/// <returns>Salidas de la capa.</returns>
		public double[] Propagacion(double[] entradas)
		{
			if (Tipo == TipoCapa.Entrada)
			{
				for (int i = 0; i < Neuronas.Length; i++)
				{
					Neuronas[i].Salida = entradas[i];
				}
				return entradas;
			}
			else
			{
				double[] Salidas = new double[Neuronas.Length];
				for (int i = 0; i < Neuronas.Length; i++)
				{
					double suma = 0;
					for (int e = 0; e < entradas.Length; e++)
					{
						suma += entradas[e] * Neuronas[i].Pesos[e];
					}
					Neuronas[i].Salida = FuncionActivacion(suma + Neuronas[i].Bias);
					Salidas[i] = Neuronas[i].Salida;
				}
				return Salidas;
			}
		}

		/// <summary>
		/// Realiza la retropropagación (backpropagation).
		/// </summary>
		/// <param name="errores">Errores de la capa siguiente.</param>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje.</param>
		public void Retropropagacion(double[] errores)
		{
			for (int i = 0; i < Neuronas.Length; i++)
			{
				Neurona neurona = Neuronas[i];
				neurona.Delta = errores[i] * FuncionDeActivacionDerivada(neurona.Salida);

				if (Tipo != TipoCapa.Entrada)
				{
					for (int w = 0; w < neurona.Pesos.Length; w++)
					{
						neurona.Pesos[w] = neurona.Pesos[w] - (VariablesGlobales.TasaAprendizaje * neurona.Delta * Neuronas[i].Salida);
					}
					neurona.Bias = VariablesGlobales.TasaAprendizaje * neurona.Delta;
				}
			}
		}

		/// <summary>
		/// Método para la función de activación: FUNCION RELU, LEAKY RELU Y SIGMOIDE
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>Regresa el valor de entrada procesada con base a la función de activación.</returns>
		private double FuncionActivacion(double x)
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
		/// Método para la función de activación derivada: FUNCION RELU, LEAKY RELU Y SIGMOIDE DERIVADA
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>La entrada ya procesada por la función de activación derivada.</returns>
		private double FuncionDeActivacionDerivada(double x)
		{
			//SIGMOIDE
			//return x * (1 - x);

			//RELU
			//return x > 0 ? 1 : 0;

			//Leaky ReLU
			return x > 0 ? 1 : 0.01;

			//ELU
			//return x > 0 ? 1 : 0.01 * Math.Exp(x);
		}
	}
}