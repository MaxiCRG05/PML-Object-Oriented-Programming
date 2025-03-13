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
		readonly Random rand = new Random();

		/// <summary>
		/// Nombre de la neurona.
		/// </summary>
		public String Nombre { get; set; }

		/// <summary>
		/// Pesos de la neurona dependiendo de las entradas que reciba.
		/// </summary>
		public double[] Pesos { get; set; }

		/// <summary>
		/// Bias de la neurona.
		/// </summary>
		public double Bias { get; set; }

		/// <summary>
		/// Delta (error) de la neurona.
		/// </summary>
		public double Delta { get; set; }

		/// <summary>
		/// Salida (predicción) de la neurona.
		/// </summary>
		public double Salida { get; set; }

		/// <summary>
		/// Lista de neuronas de la capa siguiente.
		/// </summary>
		public List<Neurona> NeuronasSiguientes = new List<Neurona>();

		/// <summary>
		/// Lista de neuronas de la capa anterior.
		/// </summary>
		public List<Neurona> NeuronasAnteriores = new List<Neurona>();

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(String name, int numeroEntradas, TipoCapa tipoCapa, int[] capas)
		{
			Nombre = name;
			Pesos = new double[numeroEntradas];

			if (tipoCapa == TipoCapa.Entrada) 
			{
				for (int i = 0; i < numeroEntradas; i++)
					IniciarPesos(capas);
			}
			else if (tipoCapa == TipoCapa.Oculta)
			{
				for (int i = 0; i < numeroEntradas; i++)
					IniciarPesos(capas);
				IniciarBias();
			}
			else if (tipoCapa == TipoCapa.Salida) 
			{
				IniciarBias();
			}
		}

		/// <summary>
		/// Método para conectar una neurona con la siguiente.
		/// </summary>
		/// <param name="neuronaSiguiente">Neurona siguiente para conectarla.</param>
		public void ConectarNeuronaSiguiente(Neurona neuronaSiguiente)
		{
			NeuronasSiguientes.Add(neuronaSiguiente);
			neuronaSiguiente.NeuronasAnteriores.Add(this);
		}

		//MÉTODO DE XAVIER (GLOROT)
		private void IniciarPesos(int[] capas)
		{
			for (int i = 0; i < capas.Length - 1; i++)
			{
				int n_in = capas[i];
				int n_out = capas[i + 1];
				double limit = (double)Math.Sqrt(6.0 / (n_in + n_out));

				Pesos[i] = (double)(rand.NextDouble() * 2 * limit - limit);
					
			}
		}

		private void IniciarBias()
		{
			Bias = (double)new Random().NextDouble() - 0.5f;
		}
	}
}
