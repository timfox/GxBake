"""Every manifest rule kind is wired up end to end: a builder, the docstring, the --init example.

Merging two branches that each added a kind once dropped [[camera]]'s builder and its example
block without a conflict; this catches that class of loss for any future kind."""

import re

from gxbake.core import build


def test_every_rule_kind_has_a_builder():
	assert set(build.BUILDERS) == set(build.RULE_KINDS)


def test_every_rule_kind_is_documented_and_in_the_init_example():
	doc = build.__doc__ or ""
	example = set(re.findall(r"^# \[\[(\w+)\]\]", build.EXAMPLE_MANIFEST, re.M))
	example |= set(re.findall(r"^\[\[(\w+)\]\]", build.EXAMPLE_MANIFEST, re.M))
	for kind in build.RULE_KINDS:
		assert f"[[{kind}]]" in doc, kind
		assert kind in example, kind


def test_rule_builders_return_lists(tmp_path):
	for kind, fn in build.BUILDERS.items():
		if kind == "gi":
			continue	# needs a log callback; covered by test_gi
		rule = build.Rule(kind, [], "", {})
		assert fn(rule, tmp_path, tmp_path / "out", {}) == [], kind
