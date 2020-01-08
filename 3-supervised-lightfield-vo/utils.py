def print_summary(ep, epochs, b, batches, loss, angle_loss, translation_loss):
	print(f"Epoch: 	{ep}/{epochs}", end="    ")
	print(f"Batch:  {b}/{batches}", end="    ")
	print("Loss: 	{:.3f}".format(loss), end="    ")
	print("Angular: {:.3f}	Linear: {:.3f}".format(angle_loss, translation_loss))

def print_short_summary(loss, angle_loss, translation_loss):
	print("Loss: 	{:.3f}".format(loss), end="    ")
	print("Angular: {:.3f}	Linear: {:.3f}".format(angle_loss, translation_loss))

def print_hrule():
	print("-"*61)