/*
see LICENSE.iitm

Author : Sujay Pandit
Email id : contact.sujaypandit@gmail.com
Details : This module implements the multiplier for RISC-V. It expects the operands and funct3
arguments to be provided.
         Additionaly, this module has been parameterized for retiming. The parameter
'number_of_multiplier_stages' defines the number of regsiters placed at the inputs. This allows for
the synthesizer to retime the circuit by moving around the registers within the combo blocks. Both
FPGA and ASIC synthesizers support this feature.

--------------------------------------------------------------------------------------------------
*/

package int_multiplier;

  import DReg :: *;
  import Vector :: * ;
  `include "Logger.bsv"
  `include "mbox_parameters.bsv"

  interface Ifc_int_multiplier;
    (*always_ready*)
		method Action send(Bit#(`XLEN) in1, Bit#(`XLEN) in2, Bit#(3) funct3
                                                                `ifdef RV64, Bool word32 `endif );
		method Tuple2#(Bit#(1),Bit#(`XLEN)) receive;
	endinterface

  (*synthesize*)
  module mk_int_multiplier(Ifc_int_multiplier);
    Vector#(`number_of_multiplier_stages, Reg#(Bit#(1)))  rg_valid  <- replicateM(mkDReg(0));
    Vector#(`number_of_multiplier_stages, Reg#(Bit#(TAdd#(1, `XLEN))))  rg_op1  <- replicateM(mkReg(0));
    Vector#(`number_of_multiplier_stages, Reg#(Bit#(TAdd#(1, `XLEN))))  rg_op2  <- replicateM(mkReg(0));
    Vector#(`number_of_multiplier_stages,Reg#(Bit#(3)))     rg_fn3  <- replicateM(mkReg(0));
    Vector#(`number_of_multiplier_stages,Reg#(Bit#(TMul#(2,TAdd#(1, `XLEN))))) rg_out  <- replicateM(mkReg(0));

    `ifdef RV64
      Vector#(`number_of_multiplier_stages,Reg#(Bool))     rg_word <- replicateM(mkReg(False));
    `endif

    Wire#(Bit#(`XLEN)) wr_result <- mkWire();

    rule rl_perform_mul;
      for(Integer i=1;i<`number_of_multiplier_stages;i=i+1) begin
        rg_out[i] <=  rg_out[i-1];
        rg_valid[i] <= rg_valid[i-1];
        rg_fn3[i] <= rg_fn3[i-1];
        `ifdef RV64
          rg_word[i] <= rg_word[i-1];
        `endif
        end
    endrule


		method Action send(Bit#(`XLEN) in1, Bit#(`XLEN) in2, Bit#(3) funct3
                                                              `ifdef RV64, Bool word32 `endif );

      Bit#(1) sign1 = funct3[1]^funct3[0];
      Bit#(1) sign2 = pack(funct3[1 : 0] == 1);
      let op1 = unpack({sign1 & in1[valueOf(`XLEN) - 1], in1});
      let op2 = unpack({sign2 & in2[valueOf(`XLEN) - 1], in2});

      rg_out[0] <= signExtend(op1) * signExtend(op2);
      rg_valid[0] <= 1'b1;
      rg_fn3[0] <= funct3;
    `ifdef RV64
      rg_word[0] <= word32;

    `endif

    endmethod
		method Tuple2#(Bit#(1),Bit#(`XLEN)) receive;
      Bool lv_upperbits = unpack(|rg_fn3[`number_of_multiplier_stages-1][1:0]);
      let out = rg_out[`number_of_multiplier_stages-1];
      Bit#(`XLEN) default_out;
      if(lv_upperbits)
        default_out = pack(out)[valueOf(TMul#(2, `XLEN)) - 1 : valueOf(`XLEN)];
      else
        default_out = pack(out)[valueOf(`XLEN) - 1:0];

      `ifdef RV64
        if(rg_word[`number_of_multiplier_stages-1])
          default_out = signExtend(default_out[31 : 0]);
      `endif
      return tuple2(rg_valid[`number_of_multiplier_stages-1],default_out);
    endmethod

  endmodule
endpackage

