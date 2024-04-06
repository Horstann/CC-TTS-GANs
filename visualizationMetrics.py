"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

   
def visualization (ori_data, generated_data, analysis, save_name=None, use_plotly=True):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    if save_name is None: save_name = analysis

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        show_plot(pca_results[:,0], pca_results[:,1], pca_hat_results[:,0], pca_hat_results[:,1], save_name=save_name, use_plotly=use_plotly, anal_sample_no=anal_sample_no, plot_title='PCA plot')

    elif analysis == 'tsne':
        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        show_plot(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], save_name=save_name, use_plotly=use_plotly, anal_sample_no=anal_sample_no, plot_title='t-SNE plot')


def show_plot(real1, real2, syn1, syn2, save_name, use_plotly=True, anal_sample_no=1000, plot_title=''):
    colors=["red" for _ in range(anal_sample_no)] + ["blue" for _ in range(anal_sample_no)] 

    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = real1, y = real2,
            mode = 'markers',
            marker = dict(color = colors[:anal_sample_no], opacity = 0.2),
            name = 'Real'
        ))
        fig.add_trace(go.Scatter(
            x = syn1, y = syn2,
            mode = 'markers',
            marker = dict(color = colors[anal_sample_no:], opacity = 0.2),
            name = 'Synthetic'
        ))

        fig.update_layout(title = plot_title, xaxis = dict(title = 'x-pca'), yaxis = dict(title = 'y_pca'), legend = dict(x = 1, y = 1))
        fig.show()
        fig.write_html(f'./images/{save_name}.html')
    else:
        f, ax = plt.subplots(1)
        plt.scatter(real1, real2, 
                    c = colors[:anal_sample_no], alpha = 0.2, label = 'Real')
        plt.scatter(syn1, syn2, 
                    c = colors[anal_sample_no:], alpha = 0.2, label = 'Synthetic')

        ax.legend()
        plt.title(plot_title)
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()
        plt.savefig(f'./images/{save_name}.pdf', format="pdf")