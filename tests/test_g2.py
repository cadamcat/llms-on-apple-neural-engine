"""Protect G2 accounting boundaries with small, hardware-free counterexamples."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from g2.evidence import account, saturated, quantile, platform, native_growth
from g2.thermal import classify


class ServiceAccounting(unittest.TestCase):
    def event(self,index,arrival,end=None,status='completed'):
        row={'id':str(index),'index':index,'arrival_ns':arrival,'deadline_ns':10,
             'status':status,'deadline_violation':status!='completed' or end-arrival>10}
        if status=='completed':row.update(service_start_ns=arrival,completed_ns=end)
        return row

    def test_rejection_stays_in_all_arrival_denominator(self):
        result=account([self.event(0,0,5),self.event(1,20,status='rejected')],[0,20],0,100,120,10)
        self.assertEqual(result['deadline_violation_fraction'],.5)
        self.assertEqual(result['response_seconds_completed_only']['count'],1)
        self.assertEqual(result['success_fraction'],.5)

    def test_drain_completion_not_observation_throughput(self):
        result=account([self.event(0,90,105)],[90],0,100,120,10)
        self.assertEqual(result['completed'],1)
        self.assertEqual(result['observed_completed'],0)
        self.assertEqual(result['deadline_violations'],1)

    def test_boundary_completion_excluded(self):
        result=account([self.event(0,90,100)],[90],0,100,120,10)
        self.assertEqual(result['observed_completed'],0)
        self.assertEqual(result['deadline_violations'],0)

    def test_missing_duplicate_or_changed_arrival_rejected(self):
        a=self.event(0,0,5);b=self.event(1,20,25)
        for events in ([a],[a,dict(b,index=0,id='0')],[a,dict(b,arrival_ns=21)]):
            with self.assertRaises(ValueError):account(events,[0,20],0,100,120,10)

    def test_forged_success_label_rejected(self):
        row=self.event(0,0,status='failed');row['deadline_violation']=False
        with self.assertRaises(ValueError):account([row],[0],0,100,120,10)

    def test_saturation_cap_is_not_completed_protocol(self):
        events=[{'index':0,'id':'0','status':'completed','completed_ns':20}]
        self.assertTrue(saturated(events,0,100,120,1)['censored'])
        with self.assertRaises(ValueError):saturated(events,0,100,120,2)

    def test_linear_interpolated_tail(self):
        self.assertAlmostEqual(quantile([0,10,20],.95),19)

    def test_missing_thermal_member_is_undetermined(self):
        group={'id':'pair','members':['C','G'],'frozen_before_results':True}
        self.assertEqual(classify(group,{}, {})['status'],'undetermined')

    def test_native_growth_rejects_unexplained_stage_calls(self):
        block={'engine':'C','host':{'pid':1,'requests':2,'stage_calls':2*16*4-1},
               'checkpoints':[{'footprint_bytes':10},{'footprint_bytes':4}]}
        with self.assertRaises(ValueError):native_growth([block])
        block['host']['stage_calls']=2*16*4
        record=native_growth([block])[0]
        self.assertEqual(record['projected_retained_bytes'],2*16*15360*256*2)
        self.assertEqual(record['footprint_change_bytes'],-6)

    def test_temperature_gap_cannot_be_platform(self):
        slot={'config':{'phase':'saturated_thermal'},'observe_start_ns':0,'observe_end_ns':int(300e9)}
        self.assertEqual(platform(slot,[],[]),'undetermined')


if __name__=='__main__':unittest.main()
