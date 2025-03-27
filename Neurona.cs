using System;
using System.Collections.Generic;
using System.Drawing;
using static Perceptron_Multicapa_Colores.Capa;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Neuronas para manejar las neuronas del PML de cada capa.
	/// </summary>
	public class Neurona
	{

		/// <summary>
		/// Generador de números.
		/// </summary>
		Random rand = new Random();

		public int numPesos;

		/// <summary>
		/// Nombre de la neurona.
		/// </summary>
		public string Nombre;

		/// <summary>
		/// Pesos de la neurona dependiendo de las entradas que reciba.
		/// </summary>
		public List<double> w = new List<double>();

		/// <summary>
		/// Bias de la neurona.
		/// </summary>
		public double b;

		/// <summary>
		/// Delta (error) de la neurona.
		/// </summary>
		public double delta;

		/// <summary>
		/// Salida (predicción) de la neurona.
		/// </summary>
		public double a;

		/// <summary>
		/// Lista de neuronas de la capa siguiente.
		/// </summary>
		public List<Neurona> neuronasSiguientes = new List<Neurona>();

		/// <summary>
		/// Lista de neuronas de la capa anterior.
		/// </summary>
		public List<Neurona> neuronasAnteriores = new List<Neurona>();

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(String name, int numNeuronas ,int numeroNeuronasCapaSiguiente, TipoCapa tipoCapa)
		{
			Nombre = name;
			numPesos =  (tipoCapa == TipoCapa.Salida) ? numNeuronas : numeroNeuronasCapaSiguiente;

			switch(tipoCapa)
			{
				case TipoCapa.Entrada:
					inicializarPesos(numeroNeuronasCapaSiguiente);
					break;

				case TipoCapa.Oculta:
					inicializarPesos(numeroNeuronasCapaSiguiente);
					inicializarBias();
					break;
				case TipoCapa.Salida:
					inicializarPesos(numNeuronas);
					inicializarBias();
					break;
			}
		}

		private void inicializarPesos(int numNeuronas)
		{
			for (int i = 0; i < numNeuronas; i++)
			{
				w.Add(rand.NextDouble());
			}
		}

		private void inicializarBias()
		{
			b = rand.NextDouble();
		}
	}
}
