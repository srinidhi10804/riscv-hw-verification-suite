/*
see LICENSE.iitm

Author : Nagakaushik Moturi
Email id : ee17b111@smail.iitm.ac.in
Details : This module implements the Carry Look Ahead Adder for 2 130 bit inputs, it uses
recursive doubling technique, computes the carries by propagating them through 7 stage carry
propagation(recursive doubling)
------------------------------------------------------------------------------------------------------
*/
package CLA_130;
  import DReg :: *;
  import Vector :: * ;

  interface Ifc_CLA;
    (*always_ready*)
		method Action send(Bit#(130) a, Bit#(130) b);
		method Tuple2#(Bit#(130),Bit#(130)) receive;
	endinterface

	(*synthesize*)
	module mk_CLA (Ifc_CLA);

	  Reg#(Bit#(130)) rg_operands1   <- mkReg(0);
	  Reg#(Bit#(130)) rg_operands2   <- mkReg(0);
	  Reg#(Bit#(130)) rg_operands11   <- mkReg(0);
	  Reg#(Bit#(130)) rg_operands22   <- mkReg(0);
	  Reg#(Bit#(130)) rg_out   <- mkReg(0);

	  Vector#(130, Reg#(Bit#(2))) out <- replicateM(mkReg(0));

	  function Bit#(2) kpg (Bit#(2) a,Bit#(2) b);    //propagation of carry
	    Bit#(2) c=2'b00;
	    // 00 - kill
	    // 01 - propagate
	    // 10 - generate
	    if (a==2'b00) c = 2'b00;
	    if (a==2'b01) c = b;
	    if (a==2'b10) c = 2'b10;
	    return c;
	  endfunction

	  function Bit#(2) carry_gen (Bit#(1) a,Bit#(1) b);   //generating the carry from the input operands
	    Bit#(2) c=2'b00;
	    // 00 - kill
	    // 01 - propagate
	    // 10 - generate
	    if ((a==0)&&(b==0)) c = 2'b00;
	    if (((a==0)&&(b==1))||((a==1)&&(b==0))) c = 2'b01;
	    if ((a==1)&&(b==1)) c = 2'b10;
	    return c;
	  endfunction

	  rule rl_CLA;                      //Carry Look Ahead Adder for 130 bit inputs, implemented using recursive doubling technique

      Vector#(131, Bit#(2)) carry_1;
      Bit#(130) carry;
      Bit#(131) c1,c0;

      for (Integer i=0; i<130; i=i+1) begin
        carry_1[i+1] = carry_gen(rg_operands1[i],rg_operands2[i]); //generating the carry from the input operands
        end
      carry_1[0] = 2'b00;

      for (Integer i=0; i<131; i=i+1) begin                        //splitting the carry vector into two single bit arrays so that we can perform bitwise operations
        c1[i] = carry_1[i][1];                                     //performing bitwise operations is essential, all other type of implementations were tried and were not compiling smoothly.
        c0[i] = carry_1[i][0];
        end

      /* this is the carry propagation done in bitwise manner
         carry propagation rules:
         kill      * (kill/propagate/generate) = kill
         propagate * (kill/propagate/generate) = (kill/propagate/generate)
         generate  * (kill/propagate/generate) = generate
         these were converted into bit forms (kill : 00 propagate: 01 generate: 10) and derived sum of products form for the carry output and performed in a bitwise manner.
         the second input is carry vector shifted by n bits.
         n = 1,2,4,8,16,32,64,128
         This is because of the recursive doubling technique used in implementation of the CLA.
      */

      c1[130:1] = ((~c1[130:1])&c0[130:1]&c1[129:0])|(c1[130:1]&(~c0[130:1]));   //n=1 propagated from adjacent carries
      c0[130:1] = ((~c1[130:1])&c0[130:1]&c0[129:0]);

      c1[130:2] = ((~c1[130:2])&c0[130:2]&c1[128:0])|(c1[130:2]&(~c0[130:2]));   //n=2 propagated from second adjacent carries
      c0[130:2] = ((~c1[130:2])&c0[130:2]&c0[128:0]);

      c1[130:4] = ((~c1[130:4])&c0[130:4]&c1[126:0])|(c1[130:4]&(~c0[130:4]));   //n=4 propagated from fourth adjacent carries
      c0[130:4] = ((~c1[130:4])&c0[130:4]&c0[126:0]);

      c1[130:8] = ((~c1[130:8])&c0[130:8]&c1[122:0])|(c1[130:8]&(~c0[130:8]));   //n=8 propagated from eigth adjacent carries
      c0[130:8] = ((~c1[130:8])&c0[130:8]&c0[122:0]);

      c1[130:16] = ((~c1[130:16])&c0[130:16]&c1[114:0])|(c1[130:16]&(~c0[130:16]));   //n=16 propagated from sixteenth adjacent carries
      c0[130:16] = ((~c1[130:16])&c0[130:16]&c0[114:0]);

      c1[130:32] = ((~c1[130:32])&c0[130:32]&c1[98:0])|(c1[130:32]&(~c0[130:32]));    //n=32 propagated from 32nd adjacent carries
      c0[130:32] = ((~c1[130:32])&c0[130:32]&c0[98:0]);

      c1[130:64] = ((~c1[130:64])&c0[130:64]&c1[66:0])|(c1[130:64]&(~c0[130:64]));    //n=64 propagated from 64th adjacent carries
      c0[130:64] = ((~c1[130:64])&c0[130:64]&c0[66:0]);

      c1[130:128] = ((~c1[130:128])&c0[130:128]&c1[2:0])|(c1[130:128]&(~c0[130:128])); //n=128 propagated from 128th adjacent carries
      c0[130:128] = ((~c1[130:128])&c0[130:128]&c0[2:0]);

      for (Integer i=1; i<131; i=i+1) begin  // updating the carry vector
        carry_1[i][1] = c1[i];
        carry_1[i][0] = c0[i];
        end

      for (Integer i=0; i<130; i=i+1) begin                //finally calculating the carry bits from the kill/propagate/generate variables after the carry propagation is complete
        carry[i] = carry_1[i+1][0]|carry_1[i+1][1];
        end

      carry=carry<<1;  //carry should be shifted because it is generated for its neighbouring bit

	    rg_out <= rg_operands1^rg_operands2^carry;          //final calculation of sum

	    rg_operands11 <= rg_operands1;
      rg_operands22 <= rg_operands2;

	  endrule

	  method Action send(Bit#(130) a, Bit#(130) b);
	    rg_operands1 <= a;
      rg_operands2 <= b;
	  endmethod
	  method Tuple2#(Bit#(130),Bit#(130)) receive;

	    return tuple2(rg_out,rg_operands11+rg_operands22);   //second output is sent for testbench to verify the rg_out (can be removed for testing timing)
	  endmethod
	endmodule
endpackage


