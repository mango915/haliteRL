def up(rolledVector, nlayers=5, ret="state"):
    halite_to_player = np.range(num_players)
    directions = np.array([0, 1, 2, 3, 4])[:, np.newaxis]
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            state[0, 0] += state[np.squeeze(action == directions, axis=1), 1].sum()
        # update ships cargo, ships presence
        state[0, [1, 3]] = 0
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            state[0, 1] = state[np.squeeze(action == directions, axis=1), 1]
    elif s == 0:
        # empty cell
        state[0, [1, 3]] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s


@numba.jit(nopython=True)
def updateState(rolledSA, mapSize, nlayers=5):
    # vfunc = np.vectorize(myfunc, excluded = ['c'])
    # newState = np.apply_along_axis(up, -1, rolledSA)
    newState = np.zeros((mapSize, mapSize, nlayers - 1), dtype=np.int64)
    for k, vector in enumerate(rolledSA):
        i = k // 6
        j = k % 6

        directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
        rolledVector = vector.reshape(5, nlayers)
        state = rolledVector[:, :-1]
        action = rolledVector[:, -1:]
        s = (directions == action).sum()
        if action[0] == 5:
            # generate dropoff
            # TODO: add halite to player's amount
            # TODO: remove local halite and give it to the player
            if state[0, 2] != 1:
                state[0, 0] = 0  # remove local halite
                state[0, 2] = -1
        if s > 1:
            # collision
            # update ships cargo, ships presence
            state[0, 1] = 0
            state[0, 3] = 0
            # drop halite in the sea
            # TODO if dropoff/shipyard leave the halite to the player
            if state[0, 2] == 1 or state[0, 2] == -1:
                update = None
                # give halite to the player
            else:
                mask = action == directions
                addends = state[mask.reshape(mask.shape[0]), 1]
                addend = np.sum(addends)
                state[0, 0] += addend
        elif s == 1:
            if action[0] == 0:
                # take halite
                # add an if ispired?
                # check cargo limits
                potentialGain = round(state[0, 0] * 0.25)
                if state[0, 1] + potentialGain <= 1000:
                    state[0, 0] -= potentialGain
                    state[0, 1] += potentialGain
                else:
                    spaceLeft = 1000 - state[0, 1]
                    state[0, 0] -= spaceLeft
                    state[0, 1] = 1000
            else:
                # just arrived
                # TODO add movement cost
                # TODO add the case which in the cell there are a shipyard/dropoff
                state[0, 3] = 1
                # print(action.dtype)
                # print(directions.dtype)
                mask = action == directions
                # print(mask.dtype)
                # print(mask)
                mask = mask.astype(np.int64)
                k = mask.shape[0]
                # mask = mask.reshape(k)
                mask = np.nonzero(mask)[0][0]
                # print(mask)
                state[0, 1] = state[mask, 1]
        elif s == 0:
            # empty cell
            state[0, 1] = 0
            state[0, 3] = 0
        else:
            print(s, "  ", action)
            raise Exception("Something strange it's happening")

        newState[i, j] = state[0]
    return newState


@numba.jit(nopython=True)
def up(rolledVector, nlayers=5, ret="state"):
    directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # update ships cargo, ships presence
        state[0, 1] = 0
        state[0, 3] = 0
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            mask = action == directions
            addends = state[mask.reshape(mask.shape[0]), 1]
            addend = np.sum(addends)
            state[0, 0] += addend
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            # print(action.dtype)
            # print(directions.dtype)
            mask = action == directions
            # print(mask.dtype)
            # print(mask)
            mask = mask.astype(np.int64)
            k = mask.shape[0]
            # mask = mask.reshape(k)
            mask = np.nonzero(mask)[0][0]
            # print(mask)
            state[0, 1] = state[mask, 1]
    elif s == 0:
        # empty cell
        state[0, 1] = 0
        state[0, 3] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s
def up(rolledVector, nlayers=5, ret="state"):
    halite_to_player = np.range(num_players)
    directions = np.array([0, 1, 2, 3, 4])[:, np.newaxis]
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            state[0, 0] += state[np.squeeze(action == directions, axis=1), 1].sum()
        # update ships cargo, ships presence
        state[0, [1, 3]] = 0
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            state[0, 1] = state[np.squeeze(action == directions, axis=1), 1]
    elif s == 0:
        # empty cell
        state[0, [1, 3]] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s


@numba.jit(nopython=True)
def updateState(rolledSA, mapSize, nlayers=5):
    # vfunc = np.vectorize(myfunc, excluded = ['c'])
    # newState = np.apply_along_axis(up, -1, rolledSA)
    newState = np.zeros((mapSize, mapSize, nlayers - 1), dtype=np.int64)
    for k, vector in enumerate(rolledSA):
        i = k // 6
        j = k % 6

        directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
        rolledVector = vector.reshape(5, nlayers)
        state = rolledVector[:, :-1]
        action = rolledVector[:, -1:]
        s = (directions == action).sum()
        if action[0] == 5:
            # generate dropoff
            # TODO: add halite to player's amount
            # TODO: remove local halite and give it to the player
            if state[0, 2] != 1:
                state[0, 0] = 0  # remove local halite
                state[0, 2] = -1
        if s > 1:
            # collision
            # update ships cargo, ships presence
            state[0, 1] = 0
            state[0, 3] = 0
            # drop halite in the sea
            # TODO if dropoff/shipyard leave the halite to the player
            if state[0, 2] == 1 or state[0, 2] == -1:
                update = None
                # give halite to the player
            else:
                mask = action == directions
                addends = state[mask.reshape(mask.shape[0]), 1]
                addend = np.sum(addends)
                state[0, 0] += addend
        elif s == 1:
            if action[0] == 0:
                # take halite
                # add an if ispired?
                # check cargo limits
                potentialGain = round(state[0, 0] * 0.25)
                if state[0, 1] + potentialGain <= 1000:
                    state[0, 0] -= potentialGain
                    state[0, 1] += potentialGain
                else:
                    spaceLeft = 1000 - state[0, 1]
                    state[0, 0] -= spaceLeft
                    state[0, 1] = 1000
            else:
                # just arrived
                # TODO add movement cost
                # TODO add the case which in the cell there are a shipyard/dropoff
                state[0, 3] = 1
                # print(action.dtype)
                # print(directions.dtype)
                mask = action == directions
                # print(mask.dtype)
                # print(mask)
                mask = mask.astype(np.int64)
                k = mask.shape[0]
                # mask = mask.reshape(k)
                mask = np.nonzero(mask)[0][0]
                # print(mask)
                state[0, 1] = state[mask, 1]
        elif s == 0:
            # empty cell
            state[0, 1] = 0
            state[0, 3] = 0
        else:
            print(s, "  ", action)
            raise Exception("Something strange it's happening")

        newState[i, j] = state[0]
    return newState


@numba.jit(nopython=True)
def up(rolledVector, nlayers=5, ret="state"):
    directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # update ships cargo, ships presence
        state[0, 1] = 0
        state[0, 3] = 0
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            mask = action == directions
            addends = state[mask.reshape(mask.shape[0]), 1]
            addend = np.sum(addends)
            state[0, 0] += addend
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            # print(action.dtype)
            # print(directions.dtype)
            mask = action == directions
            # print(mask.dtype)
            # print(mask)
            mask = mask.astype(np.int64)
            k = mask.shape[0]
            # mask = mask.reshape(k)
            mask = np.nonzero(mask)[0][0]
            # print(mask)
            state[0, 1] = state[mask, 1]
    elif s == 0:
        # empty cell
        state[0, 1] = 0
        state[0, 3] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s
def up(rolledVector, nlayers=5, ret="state"):
    halite_to_player = np.range(num_players)
    directions = np.array([0, 1, 2, 3, 4])[:, np.newaxis]
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            state[0, 0] += state[np.squeeze(action == directions, axis=1), 1].sum()
        # update ships cargo, ships presence
        state[0, [1, 3]] = 0
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            state[0, 1] = state[np.squeeze(action == directions, axis=1), 1]
    elif s == 0:
        # empty cell
        state[0, [1, 3]] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s


@numba.jit(nopython=True)
def updateState(rolledSA, mapSize, nlayers=5):
    # vfunc = np.vectorize(myfunc, excluded = ['c'])
    # newState = np.apply_along_axis(up, -1, rolledSA)
    newState = np.zeros((mapSize, mapSize, nlayers - 1), dtype=np.int64)
    for k, vector in enumerate(rolledSA):
        i = k // 6
        j = k % 6

        directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
        rolledVector = vector.reshape(5, nlayers)
        state = rolledVector[:, :-1]
        action = rolledVector[:, -1:]
        s = (directions == action).sum()
        if action[0] == 5:
            # generate dropoff
            # TODO: add halite to player's amount
            # TODO: remove local halite and give it to the player
            if state[0, 2] != 1:
                state[0, 0] = 0  # remove local halite
                state[0, 2] = -1
        if s > 1:
            # collision
            # update ships cargo, ships presence
            state[0, 1] = 0
            state[0, 3] = 0
            # drop halite in the sea
            # TODO if dropoff/shipyard leave the halite to the player
            if state[0, 2] == 1 or state[0, 2] == -1:
                update = None
                # give halite to the player
            else:
                mask = action == directions
                addends = state[mask.reshape(mask.shape[0]), 1]
                addend = np.sum(addends)
                state[0, 0] += addend
        elif s == 1:
            if action[0] == 0:
                # take halite
                # add an if ispired?
                # check cargo limits
                potentialGain = round(state[0, 0] * 0.25)
                if state[0, 1] + potentialGain <= 1000:
                    state[0, 0] -= potentialGain
                    state[0, 1] += potentialGain
                else:
                    spaceLeft = 1000 - state[0, 1]
                    state[0, 0] -= spaceLeft
                    state[0, 1] = 1000
            else:
                # just arrived
                # TODO add movement cost
                # TODO add the case which in the cell there are a shipyard/dropoff
                state[0, 3] = 1
                # print(action.dtype)
                # print(directions.dtype)
                mask = action == directions
                # print(mask.dtype)
                # print(mask)
                mask = mask.astype(np.int64)
                k = mask.shape[0]
                # mask = mask.reshape(k)
                mask = np.nonzero(mask)[0][0]
                # print(mask)
                state[0, 1] = state[mask, 1]
        elif s == 0:
            # empty cell
            state[0, 1] = 0
            state[0, 3] = 0
        else:
            print(s, "  ", action)
            raise Exception("Something strange it's happening")

        newState[i, j] = state[0]
    return newState


@numba.jit(nopython=True)
def up(rolledVector, nlayers=5, ret="state"):
    directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # update ships cargo, ships presence
        state[0, 1] = 0
        state[0, 3] = 0
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            mask = action == directions
            addends = state[mask.reshape(mask.shape[0]), 1]
            addend = np.sum(addends)
            state[0, 0] += addend
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            # print(action.dtype)
            # print(directions.dtype)
            mask = action == directions
            # print(mask.dtype)
            # print(mask)
            mask = mask.astype(np.int64)
            k = mask.shape[0]
            # mask = mask.reshape(k)
            mask = np.nonzero(mask)[0][0]
            # print(mask)
            state[0, 1] = state[mask, 1]
    elif s == 0:
        # empty cell
        state[0, 1] = 0
        state[0, 3] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s
def up(rolledVector, nlayers=5, ret="state"):
    halite_to_player = np.range(num_players)
    directions = np.array([0, 1, 2, 3, 4])[:, np.newaxis]
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            state[0, 0] += state[np.squeeze(action == directions, axis=1), 1].sum()
        # update ships cargo, ships presence
        state[0, [1, 3]] = 0
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            state[0, 1] = state[np.squeeze(action == directions, axis=1), 1]
    elif s == 0:
        # empty cell
        state[0, [1, 3]] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s


@numba.jit(nopython=True)
def updateState(rolledSA, mapSize, nlayers=5):
    # vfunc = np.vectorize(myfunc, excluded = ['c'])
    # newState = np.apply_along_axis(up, -1, rolledSA)
    newState = np.zeros((mapSize, mapSize, nlayers - 1), dtype=np.int64)
    for k, vector in enumerate(rolledSA):
        i = k // 6
        j = k % 6

        directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
        rolledVector = vector.reshape(5, nlayers)
        state = rolledVector[:, :-1]
        action = rolledVector[:, -1:]
        s = (directions == action).sum()
        if action[0] == 5:
            # generate dropoff
            # TODO: add halite to player's amount
            # TODO: remove local halite and give it to the player
            if state[0, 2] != 1:
                state[0, 0] = 0  # remove local halite
                state[0, 2] = -1
        if s > 1:
            # collision
            # update ships cargo, ships presence
            state[0, 1] = 0
            state[0, 3] = 0
            # drop halite in the sea
            # TODO if dropoff/shipyard leave the halite to the player
            if state[0, 2] == 1 or state[0, 2] == -1:
                update = None
                # give halite to the player
            else:
                mask = action == directions
                addends = state[mask.reshape(mask.shape[0]), 1]
                addend = np.sum(addends)
                state[0, 0] += addend
        elif s == 1:
            if action[0] == 0:
                # take halite
                # add an if ispired?
                # check cargo limits
                potentialGain = round(state[0, 0] * 0.25)
                if state[0, 1] + potentialGain <= 1000:
                    state[0, 0] -= potentialGain
                    state[0, 1] += potentialGain
                else:
                    spaceLeft = 1000 - state[0, 1]
                    state[0, 0] -= spaceLeft
                    state[0, 1] = 1000
            else:
                # just arrived
                # TODO add movement cost
                # TODO add the case which in the cell there are a shipyard/dropoff
                state[0, 3] = 1
                # print(action.dtype)
                # print(directions.dtype)
                mask = action == directions
                # print(mask.dtype)
                # print(mask)
                mask = mask.astype(np.int64)
                k = mask.shape[0]
                # mask = mask.reshape(k)
                mask = np.nonzero(mask)[0][0]
                # print(mask)
                state[0, 1] = state[mask, 1]
        elif s == 0:
            # empty cell
            state[0, 1] = 0
            state[0, 3] = 0
        else:
            print(s, "  ", action)
            raise Exception("Something strange it's happening")

        newState[i, j] = state[0]
    return newState


@numba.jit(nopython=True)
def up(rolledVector, nlayers=5, ret="state"):
    directions = np.array([0, 1, 2, 3, 4], dtype=np.int64).reshape(5, 1)
    rolledVector = rolledVector.reshape(5, nlayers)
    state = rolledVector[:, :-1]
    action = rolledVector[:, -1:]
    s = (directions == action).sum()
    if action[0] == 5:
        # generate dropoff
        # TODO: add halite to player's amount
        # TODO: remove local halite and give it to the player
        if state[0, 2] != 1:
            state[0, 0] = 0  # remove local halite
            state[0, 2] = -1
    if s > 1:
        # collision
        # update ships cargo, ships presence
        state[0, 1] = 0
        state[0, 3] = 0
        # drop halite in the sea
        # TODO if dropoff/shipyard leave the halite to the player
        if state[0, 2] == 1 or state[0, 2] == -1:
            update = None
            # give halite to the player
        else:
            mask = action == directions
            addends = state[mask.reshape(mask.shape[0]), 1]
            addend = np.sum(addends)
            state[0, 0] += addend
    elif s == 1:
        if action[0] == 0:
            # take halite
            # add an if ispired?
            # check cargo limits
            potentialGain = round(state[0, 0] * 0.25)
            if state[0, 1] + potentialGain <= 1000:
                state[0, 0] -= potentialGain
                state[0, 1] += potentialGain
            else:
                spaceLeft = 1000 - state[0, 1]
                state[0, 0] -= spaceLeft
                state[0, 1] = 1000
        else:
            # just arrived
            # TODO add movement cost
            # TODO add the case which in the cell there are a shipyard/dropoff
            state[0, 3] = 1
            # print(action.dtype)
            # print(directions.dtype)
            mask = action == directions
            # print(mask.dtype)
            # print(mask)
            mask = mask.astype(np.int64)
            k = mask.shape[0]
            # mask = mask.reshape(k)
            mask = np.nonzero(mask)[0][0]
            # print(mask)
            state[0, 1] = state[mask, 1]
    elif s == 0:
        # empty cell
        state[0, 1] = 0
        state[0, 3] = 0
    else:
        print(s, "  ", action)
        raise Exception("Something strange it's happening")
    if ret == "state":
        return state[0]
    elif ret == "s":
        return s
