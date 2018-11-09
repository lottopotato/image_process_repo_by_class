
// IMAGE_process_part1Dlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IMAGE_process_part1.h"
#include "IMAGE_process_part1Dlg.h"
#include "afxdialogex.h"
#include "math.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CIMAGEprocesspart1Dlg 대화 상자



CIMAGEprocesspart1Dlg::CIMAGEprocesspart1Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_IMAGE_PROCESS_PART1_DIALOG, pParent)
	, list_choice(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CIMAGEprocesspart1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, function_list);
	DDX_CBString(pDX, IDC_COMBO1, list_choice);
}

BEGIN_MESSAGE_MAP(CIMAGEprocesspart1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_CBN_SELENDOK(IDC_COMBO1, &CIMAGEprocesspart1Dlg::OnSelendokCombo1)
END_MESSAGE_MAP()


// CIMAGEprocesspart1Dlg 메시지 처리기

BOOL CIMAGEprocesspart1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	ShowWindow(SW_MAXIMIZE);

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	int m_width = 512;
	int m_height = 512;
	int m_size = m_width * m_height;

	FILE *imageFile;
	unsigned char *src_image = new unsigned char[m_size];
	x_image.Create(m_width, m_height, 32);

	if (0 == fopen_s(&imageFile, "512_512.raw", "rb")) {
		fread(src_image, m_size, 1, imageFile);
		// 0. vanilla
		Basic_image(src_image, m_width, m_height, false);
		// 0. histogram
		Histogram_graph(x_hist_image, src_image, m_width, m_height, true, true);

		// Arithmetic
		// 1. add pixel
		//ArithmeticPixel(src_image, m_width, m_height, 50, _T("Add"));
		// 2. Sub pixel
		//ArithmeticPixel(src_image, m_width, m_height, 50, _T("Sub"));
		// 3. Mul pixel
		//ArithmeticPixel(src_image, m_width, m_height, 2.5, _T("Mul"));
		// 4. Div pixel
		//ArithmeticPixel(src_image, m_width, m_height, 1.3, _T("Div"));

		// logic
		// 1. and
		//LogicPixel(src_image, m_width, m_height, _T("And"));
		// 2. or
		//LogicPixel(src_image, m_width, m_height, _T("Or"));
		// 3. xor
		//LogicPixel(src_image, m_width, m_height, _T("Xor"));

		// gamma correction
		//GammaCorrection(src_image, m_width, m_height, 0.5);

		// negertive invert
		//Negertive_invert(src_image, m_width, m_height);

		// posterizing
		//Posterizing(src_image, m_width, m_height, 8);

		// binarization
		//Binarization(src_image, m_width, m_height, 160);

		// highright range
		//Highright_range(src_image, m_width, m_height, 100, 200);

		// up or down sampling
		// 1. upSampling
		//Sampling(src_image, _T("up"), m_width, m_height, 2);
		// 2. downSampling
		//Sampling(src_image, _T("down"), m_width, m_height, 2);
		
		// histogram process
		// 1. stretching
		//Histogram_process(src_image, m_width, m_height, _T("stretching"), 255, 0);
		// 2. end-in stretching
		//Histogram_process(src_image, m_width, m_height, _T("end_in"), 50, 200);
		// 3. equalization
		//Histogram_process(src_image, m_width, m_height, _T("equalization"), 0, 0);
		// 4. specification -- 확인.
		//Histogram_Specification(src_image, dest_image, m_width, m_height, _T("Specification"));

		// convolution
		// 1. embossing
		Image_convolution_process(src_image, m_width, m_height, _T("Embossing"));

		delete[] src_image;
		fclose(imageFile);
	}
	
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CIMAGEprocesspart1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CIMAGEprocesspart1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		//CDialogEx::OnPaint();
		CPaintDC dc(this);
		x_image.Draw(dc, 10, buffer);
		if (x_hist_image != 0) {
			int x_hist_loc = buffer + x_image.GetHeight() - x_hist_image.GetHeight();
			x_hist_image.Draw(dc, 10, x_hist_loc);
			dc.TextOutW(10, x_hist_loc, _T("Histogram_graph"));
		}
		if (y_image != 0) {
			y_image.Draw(dc, 540, buffer);
			dc.TextOutW(540, 10, result);
		}
		if (y_hist_image != 0) {
			int y_hist_loc = buffer + y_image.GetHeight() - y_hist_image.GetHeight();
			y_hist_image.Draw(dc, 540, y_hist_loc);
			dc.TextOutW(540, y_hist_loc, _T("Histogram_graph"));
		}
		
		
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CIMAGEprocesspart1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CIMAGEprocesspart1Dlg::Basic_image(unsigned char *src_image, int width, int height, boolean save)
{
	// TODO: 여기에 구현 코드 추가.
	CString name = _T("original");
	int image_size = width * height;
	unsigned char *dest_image = _toPixel(src_image, image_size);
	::SetBitmapBits(x_image, image_size * 4, dest_image);
	delete[] dest_image;
	if (save == true) {
		x_image.Save(_T("result/") + name + _T(".jpg"));
	}
}


unsigned char CIMAGEprocesspart1Dlg::clamping(int pixel)
{
	// TODO: 여기에 구현 코드 추가.
	if (pixel > 255){
		pixel = 255;
	}
	else if (pixel < 0) {
		pixel = 0;
	}
	return pixel;
}


int CIMAGEprocesspart1Dlg::ArithmeticPixel(unsigned char *src_image, int width, int height,
	double meticPixel, CString mode)
{
	// TODO: 여기에 구현 코드 추가.
	result.Format(mode + _T("%.2f"), meticPixel);
	unsigned char pixel = 0;
	int image_size = width * height;
	
	for (int i = 0; i < image_size; i++) {
		pixel = src_image[i];
		if (mode == "Add")
			src_image[i] = clamping(pixel + meticPixel);
		else if (mode == "Sub")
			src_image[i] = clamping(pixel - meticPixel);
		else if (mode == "Mul")
			src_image[i] = clamping(pixel * meticPixel);
		else if (mode == "Div")
			src_image[i] = clamping(pixel / meticPixel);
		else
			return 0;
	}
	Create_image(y_image, src_image, width, height, result, true);
	return 0;
}

int CIMAGEprocesspart1Dlg::LogicPixel(unsigned char *src_image, int width, int height,
	CString mode)
{
	// TODO: 여기에 구현 코드 추가.
	result = mode;
	int image_size = width * height;
	// mask image
	unsigned char *mask_image = new unsigned char[image_size];
	for (int i = 0; i < image_size; i++) {
		if (i > 10000 && i < 100000)
			mask_image[i] = 255;
		else
			mask_image[i] = 0;
	}
	for (int i= 0; i < image_size; i++) {
		if (mode == "And") {
			src_image[i] = clamping(src_image[i] & mask_image[i]);
		}
		else if (mode == "Or") {
			src_image[i] = clamping(src_image[i] | mask_image[i]);
		}
		else if (mode == "Xor") {
			src_image[i] = clamping(src_image[i] ^ mask_image[i]);
		}
		else
			return 0;
	}
	Create_image(y_image, src_image, width, height, result, true);
	
	return 0;
}

int CIMAGEprocesspart1Dlg::GammaCorrection(unsigned char *src_image, int width, int height,
	double gamma_val)
{
	result.Format(_T("Gamma Correction_%.2f"),gamma_val);
	int image_size = width * height;
	for (int i = 0; i < image_size; i++) {
		// gamma output = 256 * pow ([input/256], r)
		src_image[i] = clamping(256 * pow((float)src_image[i] / 256, gamma_val));
	}
	Create_image(y_image, src_image, width, height, result, true);

	return 0;
}


unsigned char* CIMAGEprocesspart1Dlg::_toPixel(unsigned char* src_image, int image_size)
{
	// TODO: 여기에 구현 코드 추가.
	unsigned char *dest_image = new unsigned char[image_size * 4];

	unsigned char *src_pos = src_image;
	unsigned char *dest_pos = dest_image;
	
	unsigned char pixel = 0;
	for (int i = 0; i < image_size; i++) {
		pixel = *src_pos;
		for (int j = 0; j < 3; j++) {
			*dest_pos++ = pixel;
		}
		*dest_pos++ = 0xFF;
		*src_pos++;
	}
	return dest_image;
}


int CIMAGEprocesspart1Dlg::Save_image(CString name)
{
	// TODO: 여기에 구현 코드 추가.
	CString path = _T("result/") + name + _T(".jpg");
	y_image.Save(path);
	return 0;
}

int CIMAGEprocesspart1Dlg::Negertive_invert(unsigned char* src_image, int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	result = _T("Negertive_invert");
	// negertive : 255 - pixel
	for (int i = 0; i < width*height; i++) {
		src_image[i] = 255 - src_image[i];
	}
	Create_image(y_image, src_image, width, height, result, true);
	return 0;
}


int CIMAGEprocesspart1Dlg::Posterizing(unsigned char* src_image, int width, int height, int post_value)
{
	// TODO: 여기에 구현 코드 추가.
	result.Format(_T("Posterizing_%d"), post_value);
	int value_range = (int)(256 / post_value);
	for (int i = 0; i < width*height; i++) {
		for (int j = 0; j < post_value; j++) {
			if (value_range * j <= src_image[i] && src_image[i] <= value_range * (j + 1))
				src_image[i] = (unsigned char)(value_range * j);
		}
	}
	Create_image(y_image, src_image, width, height, result, true);
	return 0;
}


int CIMAGEprocesspart1Dlg::Binarization(unsigned char* src_image, int widht, int height, int binary_value)
{
	// TODO: 여기에 구현 코드 추가.
	result.Format(_T("Binarizaion_value_%d"), binary_value);
	for (int i = 0; i < widht*height; i++) {
		if (binary_value <= src_image[i])
			src_image[i] = 255;
		else
			src_image[i] = 0;
	}
	Create_image(y_image, src_image, widht, height, result, true);
	return 0;
}


int CIMAGEprocesspart1Dlg::Highright_range(unsigned char* src_image, int width, int height, int range_min, int range_max)
{
	// TODO: 여기에 구현 코드 추가.
	result.Format(_T("highright_range_%d_%d"), range_min, range_max);
	for (int i = 0; i < width*height; i++) {
		if (range_min <= src_image[i] && src_image[i] <= range_max)
			src_image[i] = 255;
	}
	Create_image(y_image, src_image, width, height, result, true);
	return 0;
}


int CIMAGEprocesspart1Dlg::Sampling(unsigned char* src_image, CString mode, int m_width, int m_height, int sampling_val)
{
	// TODO: 여기에 구현 코드 추가.
	if (mode == "up") {
		int r_width = (int)(m_width*sampling_val);
		int r_height = (int)(m_height*sampling_val);
		result.Format(_T("up_sampling_x%d"), sampling_val);
		unsigned char* up_image = new unsigned char[r_width*r_height];
		for (int i = 0; i < m_width; i++) {
			for (int j = 0; j < m_height; j++) {
				for (int k = 0; k < sampling_val*sampling_val; k++) {
					up_image[i*sampling_val*r_width + j*sampling_val + k] = src_image[i*m_width + j];
				}
			}
		}
		Create_image(y_image, up_image, r_width, r_height, result, true);
	}
	else if (mode == "down") {
		int r_width = (int)(m_width/sampling_val);
		int r_height = (int)(m_height/sampling_val);
		result.Format(_T("down_sampling_d%d"), sampling_val);
		unsigned char* down_image = new unsigned char[r_width*r_height];
		for (int i = 0; i < r_width; i++) {
			for (int j = 0; j < r_height; j++) {
				for (int k = 0; k < sampling_val*sampling_val; k++) {
					down_image[i*r_width + j] = src_image[i*sampling_val * m_width + j*sampling_val];
				}
			}
		}
		Create_image(y_image, down_image, r_width, r_height, result, true);
	}
	else{
		result = _T("mode must is up or down");
		return 0;
	}
	
	return 0;
}


double* CIMAGEprocesspart1Dlg::Histogram_graph(CImage& image, unsigned char* src_image, int m_width, int m_height, boolean print, boolean normalize)
{
	// TODO: 여기에 구현 코드 추가.
	// histogram graph image
	unsigned char* hist_graph = new unsigned char[256 * 256 +(256*15)];
	// hist array
	double hist[256];

	for (int i = 0; i < 256; i++)
		hist[i] = 0;
	for (int i = 0; i < m_width*m_height; i++)
		hist[(int)src_image[i]]++;
	// array normarization
	if (normalize == true) {
		int MIN = 255;
		int MAX = 0;
		for (int i = 0; i < 256; i++) {
			if (hist[i] < MIN)
				MIN = hist[i];
			if (MAX < hist[i])
				MAX = hist[i];
		}
		for (int i = 0; i < 256; i++) {
			hist[i] = (unsigned char)((hist[i] - MIN) * 255 / (MAX - MIN));
		}
	}
	// array to graph image
	for (int i = 0; i < 256 * 256; i++) {
		hist_graph[i] = 255;
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < hist[i]; j++) {
			hist_graph[256 * (256 - j - 1) + i] = 0;
		}
	}
	// histogram color
	for (int i = 256; i < 256 + 5; i++) {
		for (int j = 0; j < 256; j++) {
			hist_graph[256 * i + j] = 255;
		}
	}
	for (int i = 256 + 5; i<256 + 15; i++) {
		for (int j = 0; j<256; j++) {
			hist_graph[256 * i + j] = j;
		}
	}
	if(print == true)
		Create_image(image, hist_graph, 256, 256+15, _T("Histogram_graph"), false);
	delete[] hist_graph;
	
	for (int i = 0; i < 256; i++) {
		x_HIST[i] = hist[i];
	}
	
	return hist;
}

int CIMAGEprocesspart1Dlg::Create_image(CImage& image, unsigned char* src_image, int width, int height, CString name, boolean save)
{
	// TODO: 여기에 구현 코드 추가.
	unsigned char *dest_image = _toPixel(src_image, width*height);
	image.Create(width, height, 32);
	::SetBitmapBits(image, width * height * 4, dest_image);
	delete[] dest_image;
	if (save == true)
		Save_image(name);
	return 0;
}


int CIMAGEprocesspart1Dlg::Histogram_process(unsigned char* src_image, int width, int height,
	CString mode, int low, int high)
{
	// TODO: 여기에 구현 코드 추가.
	if (mode == "stretching") {
		// stretching : new pixel = (old pixel - low) * 255 / (high - low)
		low = 255;
		high = 0;
		for (int i = 0; i < width*height; i++) {
			if (low > src_image[i])
				low = src_image[i];
			if (high < src_image[i])
				high = src_image[i];
			
		}
		for (int i = 0; i < width*height; i++) {
			src_image[i] = (src_image[i] - low) * 255 / (high - low);
		}
	}
	else if (mode == "end_in") {
		//	end-in : if new pixel<=low, _, high<=new pixel
		//			 then new pixel = 0, equal stretching process, 255 
		for (int i = 0; i < width* height; i++) {
			if (src_image[i] <= low)
				src_image[i] = 0;
			else if (high <= src_image[i])
				src_image[i] = 255;
			else
				src_image[i] = (src_image[i] - low) * 255 / (high - low);

		}
	}
	else if (mode == "equalization") {
		// equalization : sum[i] = sigma[0:i](hist[j])
		//				  pixel = sum[i] * max(i) / N 
		Histogram_equalizer(src_image, width, height, true);
		
	}
	result = _T("histogram_") + mode;
	Histogram_graph(y_hist_image, src_image, width, height, true, true);
	Create_image(y_image, src_image, width, height, result, true);
	return 0;
}


double* CIMAGEprocesspart1Dlg::Histogram_equalizer(unsigned char* src_image, int width, int height, boolean print)
{
	// TODO: 여기에 구현 코드 추가.
	double hist_sum[256];
	double sum = 0;
	int temp = 0;
	for (int i = 0; i < 256; i++)
		x_HIST[i] = 0;

	for (int i = 0; i < width*height; i++)
		x_HIST[(int)src_image[i]]++;

	for (int i = 0; i < 256; i++) {
		sum += x_HIST[i];
		hist_sum[i] = sum;
	}
	if (print == true) {
		for (int i = 0; i < width*height; i++) {
			temp = src_image[i];
			src_image[i] = (unsigned char)(hist_sum[temp] * 255 / (width*height));
		}
	}
	else {
		for (int i = 0; i < 256; i++) {
			x_HIST[i] = hist_sum[i];
		}
	}
	return hist_sum;
}



void CIMAGEprocesspart1Dlg::OnSelendokCombo1()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	int nIndex = function_list.GetCurSel();
	int nCount = function_list.GetCount();
	if ((nIndex != LB_ERR) && (nCount > 1))
	{
		if (++nIndex < nCount)
			function_list.SetCurSel(nIndex);
		else
			function_list.SetCurSel(0);
	}
	list_index = nIndex;

	int m_width = 512;
	int m_height = 512;
	int m_size = m_width * m_height;

	FILE *imageFile;
	unsigned char *src_image = new unsigned char[m_size];

	if (0 == fopen_s(&imageFile, "512_512.raw", "rb")) {
		fread(src_image, m_size, 1, imageFile);
		ArithmeticPixel(src_image, m_width, m_height, 50, _T("Add"));

		delete[] src_image;
		fclose(imageFile);
	}

}


int CIMAGEprocesspart1Dlg::Histogram_Specification(unsigned char* src_image, unsigned char* dest_image, int width, int height, CString mode)
{
	// TODO: 여기에 구현 코드 추가.
	double* hist_sum;
	hist_sum = Histogram_equalizer(src_image, width, height, true);
	
	return 0;
}


int CIMAGEprocesspart1Dlg::Image_convolution_process(unsigned char* src_image, int width, int height, CString mode)
{
	// TODO: 여기에 구현 코드 추가.
	// 1. Embosing
	if (mode == _T("Embossing")) {
		//Histogram_equalizer(src_image, width, height, true);
		double EmboMask[3][3] = { {-1., 0., 0.}, {0.,0.,0.}, {0.,0.,1.} };
		double** m_temp_image;
		m_temp_image = Convolution_with_3kernel(src_image, EmboMask, width, height);
		m_temp_image = Embossing(m_temp_image, width, height);
		m_temp_image = Double_onScale(m_temp_image, width, height);
		src_image = _2D_arr_to_1D_arr(m_temp_image, width, height);
		delete[] m_temp_image;
		
		Histogram_graph(y_hist_image, src_image, width, height, true, true);
		
		Create_image(y_image, src_image, width, height, mode, true);
		result = mode;
		
		

	}

	return 0;
}


double** CIMAGEprocesspart1Dlg::Image2Memory(int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	double **temp;
	int i, j;
	temp = new double *[height];
	for (i = 0; i < height; i++) {
		temp[i] = new double[width];
	}
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			temp[i][j] = 0.0;
		}
	}
	return temp;
}


double** CIMAGEprocesspart1Dlg::Convolution_with_3kernel(unsigned char* src_image, double Mask[3][3],
	int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	int i, j, n, m;
	double **input_image, **output_image, S = 0.0;

	input_image = Image2Memory(height + 2, width + 2);
	output_image = Image2Memory(height, width);

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			input_image[i + 1][j + 1] = (double)src_image[i * width + j];
		}
	}

	// convolution
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					S += Mask[n][m] * input_image[i + n][j + m];
				}
			}
			// save S to output image
			output_image[i][j] = S;
			// zero to sum(S)
			S = 0.0;
		}
	}
	return output_image;
}


double** CIMAGEprocesspart1Dlg::Embossing(double** src_image, int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			src_image[i][j] += 128;
			src_image[i][j] = clamping(src_image[i][j]);
			
		}
	}
	return src_image;
}


unsigned char* CIMAGEprocesspart1Dlg::_2D_arr_to_1D_arr(double** src_image, int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	unsigned char* dest = new unsigned char[width * height];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dest[i * width + j] = src_image[i][j];
		}
	}
	return dest;
}


double** CIMAGEprocesspart1Dlg::Double_onScale(double** src_image, int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	int i, j;
	double min, max;
	
	min = max = src_image[0][0];

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (src_image[i][j] <= min)
				min = src_image[i][j];
		}
	}
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (src_image[i][j] >= max)
				max = src_image[i][j];
		}
	}

	max = max - min;
	
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			src_image[i][j] = (src_image[i][j] - min) * (255. / max);
		}
	}

	return src_image;
}
