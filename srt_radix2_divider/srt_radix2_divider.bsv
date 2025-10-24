/*
see LICENSE.iitm

Author : Rahul Bodduna, Shalender Kumar
Email id : rahulbodduna@gmail.com, cs18m050@smail.iitm.ac.in
Details : The algorithm is implemented from Atkins.pdf. An example for radix 2 can be found in example.pdf. The efficient way to
select quotients is implemented from quotient_digit.pdf. To convert the quotients from redundant binary (generated as part of algorithm
to normal binary on-the-fly conversion is used - implemented from on_the_fly.pdf

This is a variable latency algorithm. The number of cycles this division takes is the difference in lead zeros of dividend and divisor.

--------------------------------------------------------------------------------------------------
*/

package srt_radix2_divider;
`include "defined_parameters.bsv"

interface IFC_sdivider#(numeric type div_width);
	method Action ma_input_operands(Bit#(div_width) dividend, Bit#(div_width) input_divisor, Bit#(4) opcode, Bit#(3) funct3);
	method ActionValue#(Tuple2#(Bit#(1),Bit#(div_width))) mav_result;
	method Action ma_set_flush(bit c);
endinterface


module mksdivider(IFC_sdivider#(div_width))
	provisos(
		Add#(1, div_width, abs_div_width),
		Add#(1, msb_div_width, div_width),
		Add#(a__, 1, div_width),
		Add#(b__, 32, div_width),
		Add#(c__, 1, div_width_bits),
		Mul#(2, div_width, d_div_width),
		Log#(div_width, div_width_bits));

	let v_div_width = valueOf(div_width);
	let v_msb_div_width = valueOf(msb_div_width);
	let v_abs_div_width = valueOf(abs_div_width);
	let v_div_width_bits = valueOf(div_width_bits);
	let v_d_div_width = valueOf(d_div_width);

function Bit#(abs_div_width) compliment2(Bit#(abs_div_width) input_);
	Bit#(abs_div_width) result = signExtend(1'b1);
	bit carry = 1;
	bit new_carry = 1;
	result = input_^result;
	for(Integer i = 0; i < v_div_width; i = i+1) begin
		new_carry = carry;
		carry = result[i]&carry;
		result[i] = result[i]^new_carry;
	end
	return result;
endfunction

	Reg#(Bit#(abs_div_width)) rg_divisor[2] <- mkCReg(2,0);
	Reg#(Bit#(abs_div_width)) rg_remainder[2] <- mkCReg(2,0);
	Reg#(Bit#(div_width)) rg_quotient_a[2] <- mkCReg(2,0);
	Reg#(Bit#(div_width)) rg_quotient_b[2] <- mkCReg(2,0);
	Reg#(Bit#(div_width_bits)) rg_shift_divisor <- mkReg(0);
	Reg#(Bit#(1)) rg_rem_sign <- mkReg(0);
	Reg#(bit) rg_negative_quotient <- mkReg(0);
	Reg#(bit) rg_div_rem <- mkReg(0);
	Reg#(Bit#(2)) rg_en_divider <- mkReg(0);
	Reg#(bit) rg_last_quotient_bit <- mkReg(0);
	Reg#(Bit#(div_width)) rg_temp_dividend <- mkReg(0);
	Reg#(Bit#(div_width)) rg_temp_divisor <- mkReg(0);
	Reg#(Bit#(div_width_bits)) rg_cycle_counter <- mkReg(0);
	Reg#(Bool) rg_sign <- mkReg(False);
	Reg#(Bool) rg_final_cycle <- mkReg(False);

	Reg#(Bit#(1)) rg_temp_div_type <- mkReg(0);
	Reg#(bit) rg_temp <- mkReg(0);

	Wire#(Tuple2#(Bit#(1),Bit#(div_width)))  wr_wire1 <- mkDWire(tuple2(0,0));


//************************************************ rule divide **********************************************************

rule rl_divide(rg_en_divider == 1);
	Bit#(div_width) bit_mask_a = rg_quotient_a[1];
	Bit#(div_width) bit_mask_b = rg_quotient_b[1];
	Bit#(div_width_bits) cycle_counter = rg_cycle_counter;
	cycle_counter = cycle_counter + 1;
	bit_mask_b[fromInteger(v_msb_div_width)-rg_cycle_counter] = 1;
	Bit#(abs_div_width) divider = 0;
	Bit#(abs_div_width) shifted_remainder = rg_remainder[1];
	bit lv_last_quotient_bit = 0;
	if(rg_remainder[1][v_abs_div_width-1] == 0)
	begin
		if(rg_remainder[1][v_abs_div_width-2] == 1)
		begin
			divider = compliment2(rg_divisor[1]);
			bit_mask_b = bit_mask_a;
			bit_mask_a[fromInteger(v_msb_div_width)-rg_cycle_counter] = 1;
		end
	end

	else if(rg_remainder[1][v_abs_div_width-1] == 1)
	begin
		if(rg_remainder[1][v_abs_div_width-2] == 0)
	begin
			divider = rg_divisor[1];
			bit_mask_a = bit_mask_b;
			bit_mask_b[fromInteger(v_msb_div_width) - rg_cycle_counter] = 0;
			lv_last_quotient_bit = 1;
		end
	end
	if(rg_final_cycle)
	begin
		if(shifted_remainder[v_div_width]==1)
		begin
			divider = rg_divisor[1];
			rg_quotient_a[1] <= rg_quotient_b[1];
		end
		else
			divider = 0;
	end
	shifted_remainder = shifted_remainder + divider;
	if(rg_final_cycle)
	begin
		rg_final_cycle <= False;
		rg_remainder[1] <= shifted_remainder >> rg_shift_divisor;
		rg_en_divider <= 2;
	end
	else if(rg_cycle_counter == fromInteger(v_msb_div_width))
	begin
		rg_remainder[1] <= shifted_remainder;
		rg_final_cycle <= True;
		rg_last_quotient_bit <= lv_last_quotient_bit;
	end
	else
	begin
		rg_remainder[1] <= shifted_remainder << 1;
	end
	if(!rg_final_cycle) begin
		rg_quotient_a[1] <= bit_mask_a;
		rg_quotient_b[1] <= bit_mask_b;
		rg_cycle_counter <= cycle_counter;
	end
endrule
rule rl_extra(rg_en_divider == 3);
	Bit#(1)			 lv_div_or_rem=0;
	Bit#(1) lv_release = 1;
	Bit#(2) lv_en_divider = 1;
	Bit#(1) lv_rem_sign = 0;
	Bit#(1) lv_div_len = 0;
	let lv_div_type = rg_temp_div_type;
	let dividend = rg_temp_dividend;
	let divisor = rg_temp_divisor;
	rg_quotient_a[1] <= 0;
	lv_release = 0;
	if(lv_div_type[0]==0) begin
		if(dividend[v_msb_div_width]==1) begin
			lv_rem_sign = 1;
	  		dividend = compliment2({1'b1,dividend})[v_msb_div_width:0];
			if(divisor[v_msb_div_width]==1) begin
	  			divisor = compliment2({1'b1,divisor})[v_msb_div_width:0];
				rg_sign <= False;
			end
			else begin
				rg_sign <= True;
			end
		end
		else begin
			if(divisor[v_msb_div_width]==1) begin
	  			divisor = compliment2({1'b1,divisor})[v_msb_div_width:0];
				rg_sign <= True;
			end
			else
			rg_sign <= False;
		end
	end
	else begin
		rg_sign <= False;
	end
	if(dividend < divisor) begin
		lv_en_divider = 2;
		lv_release = 1;
	 	rg_remainder[1] <= {1'b0,dividend};
	end
	else begin
		let shift_dividend = countZerosMSB(dividend);
		let shift_divisor = countZerosMSB(divisor);
		rg_shift_divisor <= pack(shift_divisor)[v_div_width_bits-1:0];
		rg_divisor[1] <= {1'b0,divisor << shift_divisor};
		rg_remainder[1] <= {1'b0,dividend << shift_dividend};
		rg_cycle_counter <= pack(shift_dividend-shift_divisor-1)[v_div_width_bits-1:0];
	end
	rg_en_divider <= lv_en_divider;
	rg_rem_sign <= lv_rem_sign;
endrule

rule rl_result(rg_en_divider ==2);			//this rule writes result to wire
	Bit#(div_width) lv_out = 0;
	Bit#(abs_div_width) quotient;
	rg_en_divider <= 0;
	if(rg_sign)
		quotient = compliment2({1'b1,rg_quotient_a[1]});
	else quotient = {1'b0,rg_quotient_a[1]};

	if(rg_div_rem == 0)
		lv_out = quotient[v_msb_div_width:0];
	else begin
		if(rg_rem_sign == 1)
		begin
			let x = compliment2({1'b0,rg_remainder[1][v_msb_div_width:0]});
			lv_out = x[v_msb_div_width:0];
		end
		else
			lv_out = rg_remainder[1][v_msb_div_width:0];
	end
//	$display($time,"result in final rule %h",lv_out);
	wr_wire1 <= tuple2(1,lv_out);
endrule

method Action ma_input_operands(Bit#(div_width) dividend, Bit#(div_width) divisor, Bit#(4) opcode , Bit#(3) funct3) if(rg_en_divider == 0);
		Bit#(1)          lv_div_type=0;
		Bit#(1)			 lv_div_or_rem=0;
		Bit#(1) lv_release = 1;
		Bit#(2) lv_en_divider = 2;
		Bit#(1) lv_rem_sign = 0;
		Bit#(1) lv_div_len = 0;
		Bit#(div_width) caseK = 0;
		caseK[v_msb_div_width] = 1;
		rg_quotient_b[1] <= 0;
/*	case(div_name)
		DIV: begin
			lv_div_type   = 0;
			lv_div_or_rem = 0;
		end
		DIVU: begin
			lv_div_type   = 1;
			lv_div_or_rem = 0;
		end
		REM : begin
			lv_div_type   = 0;
			lv_div_or_rem = 1;
		end
		REMU : begin
			lv_div_type   = 1;
			lv_div_or_rem = 1;
		end
		DIVW: begin
			lv_div_type   = 0;
			lv_div_or_rem = 0;
			lv_div_len = 1;
		end
		DIVUW: begin
			lv_div_type   = 1;
			lv_div_or_rem = 0;
			lv_div_len = 1;
		end
		REMW : begin
			lv_div_type   = 0;
			lv_div_or_rem = 1;
			lv_div_len = 1;
		end
		REMUW : begin
			lv_div_type   = 1;
			lv_div_or_rem = 1;
			lv_div_len = 1;
		end
 	endcase*/
		lv_div_len = pack(opcode == 'b1110);
		lv_div_or_rem = pack(funct3 == 'b110 || funct3 == 'b111);
		lv_div_type = pack(funct3 == 'b101 || funct3 == 'b111);
	if(lv_div_len == 1)
	begin
		caseK = 'hffffffff80000000;
		if(lv_div_type == 0)
		begin
			dividend = signExtend(dividend[31:0]);
			divisor = signExtend(divisor[31:0]);
		end
		else
		begin
			dividend = zeroExtend(dividend[31:0]);
			divisor = zeroExtend(divisor[31:0]);
		end
	end
	if (divisor=='d0) begin
		rg_quotient_a[1] <= 'd-1;
		rg_remainder[1] <= {1'b0,dividend};
		rg_sign<=False;
	end

//	else if (dividend==caseK && divisor== 'd-1 && (lv_div_type[0]==0) && lv_div_or_rem==0)
	else if (dividend==caseK && divisor== 'd-1 && (lv_div_type[0]==0))
	begin
		$display("special case signed overflow");
		rg_quotient_a[1] <= dividend;
//		rg_quotient_a[1] <= -1;
		rg_sign<=False;
	end
	else if(dividend == divisor) begin
		$display("special case dividend and divisor are equal");
		rg_quotient_a[1] <= 1;
		rg_remainder[1] <= 0;
		rg_sign<=False;
	end
	else begin
	lv_en_divider=3;
	rg_temp_dividend <= dividend;
	rg_temp_divisor <= divisor;
	rg_temp_div_type <= lv_div_type;
	/*	rg_quotient_a[1] <= 0;
		lv_en_divider = 1;
		lv_release = 0;
		if(lv_div_type[0]==0) begin
			if(dividend[v_msb_div_width]==1) begin
				lv_rem_sign = 1;
		  		dividend = compliment2({1'b1,dividend})[v_msb_div_width:0];
				if(divisor[v_msb_div_width]==1) begin
		  			divisor = compliment2({1'b1,divisor})[v_msb_div_width:0];
					rg_sign <= False;
				end
				else begin
					rg_sign <= True;
				end
			end
			else begin
				if(divisor[v_msb_div_width]==1) begin
		  			divisor = compliment2({1'b1,divisor})[v_msb_div_width:0];
					rg_sign <= True;
				end
				else
				rg_sign <= False;
			end
		end
		else begin
			rg_sign <= False;
		end
		if(dividend < divisor) begin
			lv_en_divider = 2;
			lv_release = 1;
		 	rg_remainder[1] <= {1'b0,dividend};
		end
		else begin
			let shift_dividend = countZerosMSB(dividend);
			let shift_divisor = countZerosMSB(divisor);
			rg_shift_divisor <= pack(shift_divisor)[v_div_width_bits-1:0];
			rg_divisor[1] <= {1'b0,divisor << shift_divisor};
			rg_remainder[1] <= {1'b0,dividend << shift_dividend};
			rg_cycle_counter <= pack(shift_dividend-shift_divisor-1)[v_div_width_bits-1:0];
		end */
	end
	rg_rem_sign <= lv_rem_sign;
	rg_div_rem <= lv_div_or_rem;
	rg_en_divider <= lv_en_divider;
endmethod

method ActionValue#(Tuple2#(Bit#(1),Bit#(div_width))) mav_result();
	let x = wr_wire1;
	return x;
endmethod
method Action ma_set_flush(bit c);
	if(c == 0)
	begin
		$display("flushed");
		rg_en_divider <= 0;
//		rg_temp<=1;
	end
endmethod

endmodule


interface Ifc_srt_radix2_divider;
	method Action ma_start(Bit#(`XLEN) dividend, Bit#(`XLEN) divisor, Bit#(4) opcode , Bit#(3) funct3);
	method ActionValue#(Tuple2#(Bit#(1),Bit#(`XLEN))) mav_result;
	method Action ma_set_flush(bit c);
endinterface
/*
(*synthesize*)
module mkdivider(Ifc_divider);
	IFC_sdivider#(`XLEN) divider <-mksdivider();

	method Action ma_start(Bit#(`XLEN) dividend, Bit#(`XLEN) divisor, Bit#(4) opcode , Bit#(3) funct3);
			divider.ma_input_operands(dividend, divisor, Bit#(4) opcode , Bit#(3) funct3);
	endmethod

	method ActionValue#(Bit#(`XLEN)) mav_result;
		let a <- divider.mav_result;
		return a;
	endmethod

	method Action ma_set_flush(bit c);
		divider.ma_set_flush(c);
	endmethod

endmodule
*/

//*********************************************** wrapper module **********************************************************
(*synthesize*)
//(*conflict_free="mav_result,ins_rl_divide"*)
//(*conflict_free="mav_result,ins_rl_extra"*)
//(*conflict_free="mav_result,ma_start"*)
//(*conflict_free="mav_result,ma_start"*)
//(*conflict_free="mav_result,ins_rl_extra"*)
module mk_srt_radix2_divider(Ifc_srt_radix2_divider);
	IFC_sdivider#(`XLEN) ins<-mksdivider;

	Reg#(Bit#(7)) rg_count <-mkReg(0);
	Reg#(Bit#(1)) rg_flag <- mkReg(0);
	Reg#(Bit#(1)) rg_valid<-mkReg(0);
	Reg#(Bit#(`XLEN)) rg_value<-mkReg(0);
	rule rl_count;
		let x = rg_count;
		if(x == 67) x = 0;
		else x = x+1;
		rg_count <= x;
//		if(rg_count == 0) $display($time,"rg_count is zero");
	endrule

	method Action ma_start(Bit#(`XLEN) dividend, Bit#(`XLEN) divisor, Bit#(4) opcode , Bit#(3) funct3)if(rg_count == 0);
//		$display($time,"start method called");
		ins.ma_input_operands(dividend,divisor, opcode , funct3);
	endmethod

	method ActionValue#(Tuple2#(Bit#(1),Bit#(`XLEN))) mav_result();
		Bit#(`XLEN) vlu = 0;
		Bit#(1) vld = 0;
		match{.valid ,.value} <- ins.mav_result();
		if(valid == 1)
			rg_value <= value;
		if(rg_count == 67)
		begin
			if(valid == 1)
			begin
				vld = 1;
				vlu = value;
			end
			else
			begin
				vld = 1;
				vlu = rg_value;
			end
		end
		return tuple2(vld,vlu);
	endmethod

	method Action ma_set_flush(bit c);
		ins.ma_set_flush(c);
	endmethod

endmodule
//********************************************* test bench module ************************************************
module tb_srt_radix2_divider();
	function Bit#(`XLEN) fn_compliment2(Bit#(`XLEN) lv_input);
		Bit#(`XLEN) lv_result = signExtend(1'b1);
		bit lv_carry = 1;
		bit lv_new_carry = 1;
		lv_result = lv_input^lv_result;
		for(Integer i = 0; i < `XLEN; i = i+1) begin
			lv_new_carry = lv_carry;
			lv_carry = lv_result[i]&lv_carry;
			lv_result[i] = lv_result[i]^lv_new_carry;
		end
		return lv_result;
	endfunction



	Ifc_srt_radix2_divider ifc_div <- mk_srt_radix2_divider();
	Reg#(int) rg_cycle <- mkReg(0);
//	Reg#(ALU_func) rg_op <- mkReg(DIV);
	Reg#(Bit#(4)) rg_opcode <- mkReg('b1100);
	Reg#(Bit#(3)) rg_funct3 <- mkReg('b100);
	Reg#(Bit#(6)) rg_cnt <- mkReg(0);
	rule rl_cycle;
		rg_cycle <= rg_cycle +1;
//		if(rg_cycle==100)
//			$finish(0);
	endrule
//	rule rl_stage_1(rg_cycle % 2==0);
	rule rl_stage_1;
		Bit#(`XLEN) op1 ='hab9436;
		Bit#(`XLEN) op2 ='h17ab;
		Bit#(`XLEN) dividend = 0;
		Bit#(`XLEN) divisor = 0;
		if(rg_cnt == 0)
		begin
			$display("%h, %h ",fn_compliment2(op1),fn_compliment2(op2));
			$display($time,"DIV");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 1)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 2)
		begin
//			$finish(0);
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 3)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= REM;
			rg_funct3 <= 'b110;
		end
//******************************************************************************************
		else if(rg_cnt == 4)
		begin
			$display($time,"REM");
			dividend = op1;
			divisor = op2;
//			$display("dividend %h , divisor %h ",dividend,divisor);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 5)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
//			$display("dividend %h , divisor %h ",dividend,divisor);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 6)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
//			$display("dividend %h , divisor %h ",dividend,divisor);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 7)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
//			$display("dividend %h , divisor %h ",dividend,divisor);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= DIVU;
			rg_funct3 <= 'b101;
		end
//******************************************************************************************
		else if(rg_cnt == 8)
		begin
			$display($time,"DIVU");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 9)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 10)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 11)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= REMU;
			rg_funct3 <= 'b111;
		end
//******************************************************************************************
		else if(rg_cnt == 12)
		begin
			$display($time,"REMU");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 13)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 14)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 15)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= DIVW;
			rg_funct3 <= 'b100;
			rg_opcode <= 'b1110;
		end
//******************************************************************************************
		else if(rg_cnt == 16)
		begin
			$display($time,"DIVW");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 17)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 18)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 19)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= REMW;
			rg_funct3 <= 'b110;
		end
//***************************************************************************************
		else if(rg_cnt == 20)
		begin
			$display($time,"REMW");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 21)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 22)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 23)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= DIVUW;
			rg_funct3 <= 'b101;
		end
//**************************************************************************************
		else if(rg_cnt == 24)
		begin
			$display($time,"DIVUW");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 25)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 26)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 27)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= REMUW;
			rg_funct3 <= 'b111;
		end
//*************************************************************************************
		else if(rg_cnt == 28)
		begin
			$display($time,"REMUW");
			dividend = op1;
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 29)
		begin
			dividend = fn_compliment2(op1);
			divisor = op2;
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 30)
		begin
			dividend = op1;
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
		end
		else if(rg_cnt == 31)
		begin
			dividend = fn_compliment2(op1);
			divisor = fn_compliment2(op2);
			ifc_div.ma_start(dividend,divisor,rg_opcode , rg_funct3);
			rg_cnt<=rg_cnt+1;
//			rg_op <= DIVW;
		end
		else $finish(0);
//******************************************************************************************

	endrule
	rule rl_receive;
//		$display("receive fule fired");
		match {.valid,.out} <- ifc_div.mav_result();
//		`logLevel( tb, 0, $format("Cycle %d => valid %d value %d",rg_cycle,valid,out))
		if(valid == 1)
		$display($time,"Cycle %d => valid %d value %h",rg_cycle,valid,out);
	endrule
endmodule
//**************


endpackage

