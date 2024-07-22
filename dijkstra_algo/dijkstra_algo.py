import random
import functools


no_nodes = 20
state = [random.choice([0, 1, 2]) for i in range(no_nodes)]
bottom = 0
top = no_nodes - 1

eligible_nodes = []
f = open("chosen.txt", "w")

def override_print(*args):
	pass

print = override_print

#Function that returns the lamda function 
def decorate_update(func):
	#Ensures that the wrappers functions retains the metadata from the original func
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		return lambda: func(*args, **kwargs)
	return wrapper


@decorate_update
def bottom_eligible_update():
	state[bottom] = (state[bottom] - 1) % 3


@decorate_update
def top_eligible_update():
	state[top] = (state[top-1] + 1) % 3


@decorate_update
def other_eligible_update(i, L_R):
	state[i] = state[L_R]


for _ in range(1000000):
	print(f"state: {state}")
	# check for bottom
	if (state[bottom] + 1) % 3 == state[bottom + 1]:
		eligible_nodes.append((bottom, bottom_eligible_update()))


	if state[top-1] == state[bottom] and (state[top-1] + 1) % 3 != state[top]:
		eligible_nodes.append((top, top_eligible_update()))


	for i in range(bottom+1, top):

		if (state[i] + 1) % 3 == state[i-1]:
			eligible_nodes.append((i, other_eligible_update(i, i-1)))
	
		if (state[i] + 1) % 3 == state[i+1]:
			eligible_nodes.append((i, other_eligible_update(i, i+1)))
	
	print(f"No. of eligible nodes: {len(eligible_nodes)}")
	chosen = random.sample(eligible_nodes, 1)[0]
	print(f"Chosen: {chosen[0]}")
	f.write(f"{chosen[0]} ")
	chosen[1]()
	eligible_nodes = []
	print()	
