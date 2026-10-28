"""Key-role separation: three+ general computer accounts, one dedicated lab account."""

import pytest

from blackthorn.keys_role import (KeyRoleError, computer_keys, describe, lab_key, lab_keys,
                                  overlap, require_lab_key)

COMPUTERS = ["pds_computer_aaaaaaaaaaaa1", "pds_computer_bbbbbbbbbbbb2", "pds_computer_cccccccccccc3"]
LAB = "pds_lab_dddddddddddddd4"


def env(**extra):
    base = {"PANDASTACK_API_KEY": COMPUTERS[0],
            "PANDASTACK_API_KEY_2": COMPUTERS[1],
            "PANDASTACK_API_KEY_3": COMPUTERS[2],
            "PANDASTACK_LAB_API_KEY": LAB}
    base.update(extra)
    return base


def test_computer_pool_excludes_the_dedicated_lab_key():
    keys = computer_keys(env())
    assert keys == COMPUTERS
    assert LAB not in keys


def test_lab_uses_only_its_own_credential():
    assert lab_key(env()) == LAB
    assert lab_keys(env()) == [LAB]
    assert require_lab_key(env()) == LAB


def test_lab_without_its_own_key_is_not_configured():
    """A deployment with only computer keys must not silently run the lab on them."""
    only_computers = {"PANDASTACK_API_KEY": COMPUTERS[0], "PANDASTACK_API_KEY_2": COMPUTERS[1]}
    assert lab_key(only_computers) == ""
    with pytest.raises(KeyRoleError):
        require_lab_key(only_computers)


def test_a_key_listed_for_both_roles_is_reported_and_removed_from_computers():
    mixed = env(PANDASTACK_API_KEYS=LAB + "," + "pds_computer_eeeeeeeeeeee5")
    assert overlap(mixed) == [] or overlap(mixed) == [LAB]
    keys = computer_keys(mixed)
    assert LAB not in keys
    assert "pds_computer_eeeeeeeeeeee5" in keys


def test_keys_are_never_printed_in_full():
    info = describe(env())
    blob = str(info)
    for key in COMPUTERS + [LAB]:
        assert key not in blob, "describe() must only expose fingerprints"
    assert info["computer_key_count"] == 3
    assert info["lab_configured"] is True
    assert info["lab_fingerprint"] == LAB[-4:]


def test_panda_key_pool_ignores_the_lab_key():
    from blackthorn.panda_keys import PandaKeyPool
    pool = PandaKeyPool.from_env(None, None, env())
    assert pool is not None
    assert all(state.key != LAB for state in pool.states.values())
    assert len(pool.states) == 3
