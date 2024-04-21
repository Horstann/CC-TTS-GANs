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

def dim_reduction(data_list, n_components=2, mode='pca', fit_shape=True):
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, *[len(data) for data in data_list]])
    idx = np.random.permutation(anal_sample_no)[:anal_sample_no]

    # Data preprocessing
    for i in range(len(data_list)):
        data_list[i] = np.asarray(data_list[i])
        data_list[i] = data_list[i][idx]

    no, seq_len, dim = data_list[0].shape  

    for i in range(len(data_list)):
        prep_data = np.reshape(np.mean(data_list[i][0,:,:], 1), [1,seq_len])
        # prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
        # prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        for j in range(1, anal_sample_no):
            prep_data = np.concatenate((prep_data, np.reshape(np.mean(data_list[i][j,:,:],1), [1,seq_len])))
        data_list[i] = prep_data

    res_list = None
    
    if mode=='pca':
        pca = PCA(n_components=n_components)
        pca.fit(data_list[0])
        res_list = [pca.transform(data_list[i]) for i in range(len(data_list))]
    elif mode=='tsne':
        # Do t-SNE Analysis together       
        data_final = np.concatenate(data_list, axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_res = tsne.fit_transform(data_final)
        res_list = [tsne_res[anal_sample_no*i:anal_sample_no*(i+1),:] for i in range(len(data_list))]
    else:
        raise NotImplementedError(mode)

    if fit_shape: res_list = [res.reshape(res.shape[0],-1,1) for res in res_list]
    return res_list
   
def visualization (data_list, name_list, analysis, save_name=None, use_plotly=True):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - analysis: tsne or pca
    """  
    if save_name is None: save_name = analysis
    anal_sample_no = min([1000, *[len(data) for data in data_list]])
    res_list = dim_reduction(data_list, n_components=2, mode=analysis, fit_shape=False)
    
    # Plotting
    show_plot(res_list, name_list, save_name=save_name, use_plotly=use_plotly, anal_sample_no=anal_sample_no, plot_title=f'{analysis} plot')

# def show_plot(real1, real2, syn1, syn2, save_name, use_plotly=True, anal_sample_no=1000, plot_title=''):
def show_plot(res_list, name_list, save_name, use_plotly=True, anal_sample_no=1000, plot_title=''):
    unique_colors = ["red", "blue", "green"]
    assert len(res_list)==len(name_list)
    assert len(res_list)<=len(unique_colors)
    colors = []
    for i in range(len(res_list)):
        colors.append([unique_colors[i] for _ in range(anal_sample_no)])

    if use_plotly:
        fig = go.Figure()
        for i, (name, res) in enumerate(zip(name_list, res_list)):
            fig.add_trace(go.Scatter(
                x=res[:,0], y=res[:,1],
                mode='markers',
                marker=dict(color=colors[i], opacity=0.2),
                name=name
            ))
        # fig.add_trace(go.Scatter(
        #     x = real1, y = real2,
        #     mode = 'markers',
        #     marker = dict(color = colors[:anal_sample_no], opacity = 0.2),
        #     name = 'Real'
        # ))
        # fig.add_trace(go.Scatter(
        #     x = syn1, y = syn2,
        #     mode = 'markers',
        #     marker = dict(color = colors[anal_sample_no:], opacity = 0.2),
        #     name = 'Synthetic'
        # ))

        fig.update_layout(title = plot_title, xaxis = dict(title = 'x-pca'), yaxis = dict(title = 'y_pca'), legend = dict(x = 1, y = 1))
        fig.show()
        fig.write_html(f'./images/{save_name}.html')
    else:
        f, ax = plt.subplots(1)
        for i, (name, res) in enumerate(zip(name_list, res_list)):
            plt.scatter(res[:,0], res[:,1], c=colors[i], alpha=0.2, label=name)
        # plt.scatter(real1, real2, 
        #             c = colors[:anal_sample_no], alpha = 0.2, label = 'Real')
        # plt.scatter(syn1, syn2, 
        #             c = colors[anal_sample_no:], alpha = 0.2, label = 'Synthetic')

        ax.legend()
        plt.title(plot_title)
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()
        plt.savefig(f'./images/{save_name}.pdf', format="pdf")