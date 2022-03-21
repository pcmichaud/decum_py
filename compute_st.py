	A = np.zeros((J,J))
	B = np.zeros((J,J))
	for i in data.index:
		e_i = es.loc[i,:].to_numpy()
		e_i[np.isnan(e_i)] = 0
		g_i = grad.loc[(i,),:].to_numpy()
		g_i[np.isnan(g_i)] = 0
		A += g_i.T @ g_i
		B += 



