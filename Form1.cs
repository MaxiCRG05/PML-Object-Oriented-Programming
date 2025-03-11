using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
    public partial class Form1 : Form
    {
        readonly Archivos archivos;
        readonly PML perceptronMultiCapa;

        public Color color;

        public Form1()
        {
            perceptronMultiCapa = new PML(VariablesGlobales.DatosPerceptron);

            archivos = new Archivos(VariablesGlobales.Ruta);
            archivos.BuscarArchivo(VariablesGlobales.Datos + VariablesGlobales.FormatoArchivos);

            InitializeComponent();

            if (archivos.BuscarArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos))
            {
                DialogResult dialog = MessageBox.Show($"El archivo, si se ha encontrado. ¿Deseas cargar los pesos?", caption: "Perceptron", buttons: MessageBoxButtons.YesNo);
                if (dialog == DialogResult.Yes)
                {
                    perceptronMultiCapa.CargarDatos();

                    btnProbar.Enabled = true;
                    button2.Enabled = false;
                }
            }
            else
            {
                btnProbar.Enabled = false; 
                button2.Enabled = true;    
            }
        }

        private void btnProbar_Click(object sender, EventArgs e)
        {
            double[] entradas = { color.R, color.G, color.B };

            double[] salida = perceptronMultiCapa.Propagacion(entradas);

            int clasePredicha = Array.IndexOf(salida, salida.Max());
            
            string nombreColor = VariablesGlobales.NombresColores[clasePredicha];

            label2.Text = $"{nombreColor}";
        }

        private void btnGuardar_MouseClick(object sender, MouseEventArgs e)
        {
            perceptronMultiCapa.GuardarDatos();  
        }

        private void btnLimpiarPesos_MouseClick(object sender, MouseEventArgs e)
        {
            archivos.CrearArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
        }

        private void btnImagen1_MouseClick(object sender, MouseEventArgs e)
        {
            try
            {
                pictureBox1.Image = Properties.Resources.color;
                MessageBox.Show("Imagen 1 colocada con exito.", "Imagen");
            }
            catch
            {
                MessageBox.Show("La imagen 1 no se ha podido colocar.", "Imagen");
            }
        }

        private void btnImagen2_MouseClick_1(object sender, MouseEventArgs e)
        {
            try
            {
                pictureBox1.Image = Properties.Resources.color1;
                MessageBox.Show("Imagen 2 colocada con exito.", "Imagen");
            }
            catch
            {
                MessageBox.Show("La imagen 2 no se ha podido colocar.", "Imagen");
            }
        }

        private void btnImagenPersonalizada_MouseClick(object sender, MouseEventArgs e)
        {
            try
            {

                openFileDialog1.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif";
                openFileDialog1.Title = "Seleccionar una imagen";

                if (openFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    string rutaImagen = openFileDialog1.FileName;
                    pictureBox1.Image = Image.FromFile(rutaImagen);
                    MessageBox.Show("Imagen cargada con éxito.", "Imagen");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error al cargar la imagen: " + ex.Message, "Error");
            }
        }

        private void button2_MouseClick(object sender, MouseEventArgs e)
        {
            button2.Enabled = false;
            perceptronMultiCapa.Entrenar(VariablesGlobales.Entradas, VariablesGlobales.Salidas, VariablesGlobales.Epocas);
            btnProbar.Enabled = true;

        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                Bitmap bitmap = (Bitmap)pictureBox1.Image;
                int x = e.X * bitmap.Width / pictureBox1.ClientSize.Width;
                int y = e.Y * bitmap.Height / pictureBox1.ClientSize.Height;
                color = bitmap.GetPixel(x, y);
                registroColores.Text += $"R: {color.R} \tG: {color.G} \tB: {color.B}\n";
                archivos.EscribirArchivo($"[{color.R}, {color.G}, {color.B}]", VariablesGlobales.Datos + VariablesGlobales.FormatoArchivos);
            }
            else
            {
                MessageBox.Show("No hay imagen cargada en el PictureBox.");
            }
        }

        private void button1_MouseClick(object sender, MouseEventArgs e)
        {
            registroColores.Text = "";
            label2.Text = "";
        }
    }
}
