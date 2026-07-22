// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::io::Write;
use std::path::PathBuf;

use tempfile::TempDir;

use super::{PlaceIdx, TransitionIdx};
use crate::model::{PreparedModel, PropertyAliases};

pub(super) const MINIMAL_PT_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p1">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
    </page>
  </net>
</pnml>"#;

pub(super) const COLORED_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="p1">
      <place id="P0"/>
    </page>
  </net>
</pnml>"#;

pub(super) const COLLAPSIBLE_ALL_COLORED_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="collapse-all" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="P0">
        <type><structure><usersort declaration="s1"/></structure></type>
        <hlinitialMarking><structure><all><usersort declaration="s1"/></all></structure></hlinitialMarking>
      </place>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0">
        <hlinscription><structure><all><usersort declaration="s1"/></all></structure></hlinscription>
      </arc>
      <arc id="a2" source="T0" target="P0">
        <hlinscription><structure><all><usersort declaration="s1"/></all></structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="s1" name="S">
        <cyclicenumeration>
          <feconstant id="c1" name="a"/>
          <feconstant id="c2" name="b"/>
          <feconstant id="c3" name="c"/>
        </cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

pub(super) const COLLAPSIBLE_ALL_WITH_IRRELEVANT_COLORED_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="collapse-all-relevance" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="P0">
        <type><structure><usersort declaration="s1"/></structure></type>
        <hlinitialMarking><structure><all><usersort declaration="s1"/></all></structure></hlinitialMarking>
      </place>
      <place id="Q0">
        <type><structure><usersort declaration="s1"/></structure></type>
        <hlinitialMarking><structure><all><usersort declaration="s1"/></all></structure></hlinitialMarking>
      </place>
      <transition id="T0"/>
      <transition id="T_irrelevant"/>
      <arc id="a1" source="P0" target="T0">
        <hlinscription><structure><all><usersort declaration="s1"/></all></structure></hlinscription>
      </arc>
      <arc id="a2" source="T0" target="P0">
        <hlinscription><structure><all><usersort declaration="s1"/></all></structure></hlinscription>
      </arc>
      <arc id="a3" source="Q0" target="T_irrelevant">
        <hlinscription><structure><all><usersort declaration="s1"/></all></structure></hlinscription>
      </arc>
      <arc id="a4" source="T_irrelevant" target="Q0">
        <hlinscription><structure><all><usersort declaration="s1"/></all></structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="s1" name="S">
        <cyclicenumeration>
          <feconstant id="c1" name="a"/>
          <feconstant id="c2" name="b"/>
          <feconstant id="c3" name="c"/>
        </cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

pub(super) const NUPN_PT_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="nupn-test" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p1">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
      <toolspecific tool="nupn" version="1.1">
        <size places="2" transitions="1" arcs="2"/>
        <structure units="2" root="u0" safe="true">
          <unit id="u0">
            <places/>
            <subunits>u1</subunits>
          </unit>
          <unit id="u1">
            <places>P0 P1</places>
            <subunits/>
          </unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;

pub(super) const UNFOLDED_ALIAS_PT_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="alias-test" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p1">
      <place id="Fork_0"><initialMarking><text>1</text></initialMarking></place>
      <place id="Fork_1"><initialMarking><text>1</text></initialMarking></place>
      <place id="Done"/>
      <transition id="Take_0"/>
      <transition id="Take_1"/>
      <arc id="a1" source="Fork_0" target="Take_0"/>
      <arc id="a2" source="Take_0" target="Done"/>
      <arc id="a3" source="Fork_1" target="Take_1"/>
      <arc id="a4" source="Take_1" target="Done"/>
    </page>
  </net>
</pnml>"#;

pub(super) fn write_pnml(dir: &TempDir, content: &str) {
    let path = dir.path().join("model.pnml");
    let mut f = std::fs::File::create(path).expect("create model.pnml");
    f.write_all(content.as_bytes()).expect("write model.pnml");
}

pub(super) fn mcc_input_dir(model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/mcc/2024/INPUTS")
        .join(model)
}

pub(super) fn require_mcc_input_dir(model: &str) -> Option<PathBuf> {
    let dir = mcc_input_dir(model);
    if !dir.join("model.pnml").exists() {
        eprintln!("SKIP: {model} MCC input not available");
        return None;
    }
    Some(dir)
}

pub(super) fn alias_enriched(model: &PreparedModel) -> PropertyAliases {
    let mut aliases = model.aliases.clone();
    aliases
        .place_aliases
        .insert(String::from("Fork"), vec![PlaceIdx(0), PlaceIdx(1)]);
    aliases.transition_aliases.insert(
        String::from("Take"),
        vec![TransitionIdx(0), TransitionIdx(1)],
    );
    aliases
}

pub(super) fn write_upper_bounds_alias_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>AliasModel-UpperBounds-00</id>
    <description>sum unfolded Fork places</description>
    <formula>
      <place-bound>
        <place>Fork</place>
      </place-bound>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("UpperBounds.xml"), xml).expect("write UpperBounds.xml");
}

pub(super) fn write_collapsible_all_upper_bounds_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>CollapseAll-UpperBounds-00</id>
    <description>sum all original P0 colors</description>
    <formula>
      <place-bound>
        <place>P0</place>
      </place-bound>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("UpperBounds.xml"), xml).expect("write UpperBounds.xml");
}

pub(super) fn write_collapsible_all_fireability_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>CollapseAll-ReachabilityFireability-00</id>
    <description>T0 is fireable</description>
    <formula>
      <exists-path>
        <finally>
          <is-fireable><transition>T0</transition></is-fireable>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("ReachabilityFireability.xml"), xml)
        .expect("write ReachabilityFireability.xml");
}

pub(super) fn write_collapsible_all_ctl_fireability_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>CollapseAll-CTLFireability-00</id>
    <description>T0 is fireable in the colored source</description>
    <formula>
      <exists-path>
        <finally>
          <is-fireable><transition>T0</transition></is-fireable>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("CTLFireability.xml"), xml).expect("write CTLFireability.xml");
}

pub(super) fn write_collapsible_all_ctl_cardinality_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>CollapseAll-CTLCardinality-00</id>
    <description>all original P0 colors are marked initially</description>
    <formula>
      <exists-path>
        <finally>
          <integer-le>
            <integer-constant>3</integer-constant>
            <tokens-count><place>P0</place></tokens-count>
          </integer-le>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("CTLCardinality.xml"), xml).expect("write CTLCardinality.xml");
}

pub(super) fn write_collapsible_all_ltl_fireability_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>CollapseAll-LTLFireability-00</id>
    <description>T0 is fireable in the colored source</description>
    <formula>
      <all-paths>
        <finally>
          <is-fireable><transition>T0</transition></is-fireable>
        </finally>
      </all-paths>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("LTLFireability.xml"), xml).expect("write LTLFireability.xml");
}

pub(super) fn write_collapsible_all_ltl_cardinality_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>CollapseAll-LTLCardinality-00</id>
    <description>all original P0 colors are marked initially</description>
    <formula>
      <all-paths>
        <finally>
          <integer-le>
            <integer-constant>3</integer-constant>
            <tokens-count><place>P0</place></tokens-count>
          </integer-le>
        </finally>
      </all-paths>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("LTLCardinality.xml"), xml).expect("write LTLCardinality.xml");
}

pub(super) fn write_reachability_fireability_alias_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>AliasModel-ReachabilityFireability-00</id>
    <description>some unfolded Take transition is fireable</description>
    <formula>
      <exists-path>
        <finally>
          <is-fireable><transition>Take</transition></is-fireable>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("ReachabilityFireability.xml"), xml)
        .expect("write ReachabilityFireability.xml");
}

pub(super) fn write_reachability_cardinality_alias_xml(dir: &TempDir) {
    let xml = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>AliasModel-ReachabilityCardinality-00</id>
    <description>sum unfolded Fork places in the initial state</description>
    <formula>
      <exists-path>
        <finally>
          <integer-le>
            <integer-constant>2</integer-constant>
            <tokens-count>
              <place>Fork</place>
            </tokens-count>
          </integer-le>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;
    std::fs::write(dir.path().join("ReachabilityCardinality.xml"), xml)
        .expect("write ReachabilityCardinality.xml");
}
