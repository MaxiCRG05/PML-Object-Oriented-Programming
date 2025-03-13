using System;

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
		public Capa(int numeroNeuronas, int numeroNeuronasCapaSiguiente, TipoCapa tipo, int[] capas, Capa CapaSiguiente = null)
		{
			Tipo = tipo;
			Neuronas = new Neurona[numeroNeuronas];

			if (tipo == TipoCapa.Entrada)
			{
				for (int i = 0; i < numeroNeuronas; i++)
				{
					Neuronas[i] = new Neurona($"Neurona_{i}", 0, tipo, capas);
				}
			}
			else
			{
				for (int i = 0; i < numeroNeuronas; i++)
				{
					Neuronas[i] = new Neurona($"Neurona_{i}", numeroNeuronasCapaSiguiente, tipo, capas);
				}
			}

			if (CapaSiguiente != null)
			{
				for (int i = 0; i < Neuronas.Length; i++)
				{
					for (int j = 0; j < CapaSiguiente.Neuronas.Length; j++)
					{
						CapaSiguiente.Neuronas[j].ConectarNeuronaSiguiente(Neuronas[i]);
					}
				}
			}
		}
	}
}